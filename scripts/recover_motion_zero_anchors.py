"""無風・ローター停止中の運動を補正し、T/D/Hの零点候補を再評価する。

空力fitの残差は使用しない。現行の質量・CG・センサ自己荷重・時計を使用。
動いている機体の空力と未記録の接触は分離できないため、低速窓に限定し、
別runの窓を丸ごと隠した整合性検査を通る電源区間だけ候補として出力する。
既存の停止基準は変更せず、新規基準は motion_corrected_absolute と明示する。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyulog import ULog
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a


@dataclass(frozen=True)
class Limits:
    windows_s: tuple = (6.0, 3.0)
    motor_guard_s: float = 2.0
    max_rate_deg_s: float = 5.0
    max_acceleration_deg_s2: float = 50.0
    # Original stopped-anchor consistency scales; not certified error bounds.
    wrench_scales: tuple = (0.5, 0.5, 0.5, 0.03, 0.03, 0.005)
    minimum_runs: int = 2


def predict_sensor_load(raw, cfg, sensor):
    """External wrench=0 -> expected FLU sensor load, including moving sensor.

    Invert the actual production reconstruction, so action/reaction and origin
    shifts cannot silently use a different convention from the main analysis.
    """
    zero = dict(raw, filtered_load=np.zeros_like(raw['filtered_load']))
    body, _, _, _ = a.reconstruct_body(zero, replace(cfg, bias_sensor=(0.0,) * 6), sensor)
    cg = np.asarray(body.cg_flu_m) * a.FLU_TO_FRD
    return -a.shift_wrench(zero['target'], -cg) * np.tile(a.FLU_TO_FRD, 2)


def at_cg(values, cfg, sensor):
    body, _ = a.split_calibration(cfg, sensor)
    values=np.asarray(values)
    return a.shift_wrench(np.atleast_2d(values) * np.tile(a.FLU_TO_FRD, 2),
                         np.asarray(body.cg_flu_m) * a.FLU_TO_FRD).reshape(values.shape)


def motion_bias(raw, cfg, sensor):
    return raw['filtered_load'] - predict_sensor_load(raw, cfg, sensor)


def encoder_motion(raw, cfg, seconds=1.0):
    """ULog収録前も、実測エンコーダーを局所多項式で微分する。

    未来情報を使うオフライン推定。欠損を跨がず、窓端は捨てる。
    gyroを架空に作って二重観測として投入しない。
    """
    n=len(raw['t'])
    width=max(5,round(seconds*cfg.hz)|1)
    state=np.full((n,2,3),np.nan)
    valid=np.zeros(n,bool)
    for ix in a.blocks(np.isfinite(np.c_[raw['angles'],raw['load']]).all(axis=1)):
        if len(ix)<width+2:
            continue
        for derivative in range(3):
            state[ix,:,derivative]=savgol_filter(raw['angles'][ix],width,3,
                deriv=derivative,delta=1/cfg.hz,axis=0)
        trim=max(width//2,round(cfg.smooth_seconds*cfg.hz))
        valid[ix[trim:-trim]]=True
    theta,psi=state[:,0,0],state[:,1,0]
    td,pd=state[:,0,1],state[:,1,1]
    tdd,pdd=state[:,0,2],state[:,1,2]
    return dict(raw,joint_angle=state[:,:,0],valid_motion=valid,
        rotation=a.rig_rotation(theta,psi,raw['heading'],cfg),
        omega=np.c_[np.sin(psi)*td,np.cos(psi)*td,pd],
        alpha=np.c_[np.sin(psi)*tdd+np.cos(psi)*pd*td,
                    np.cos(psi)*tdd-np.sin(psi)*pd*td,pdd],
        filtered_load=a.load_bandwidth_filter(raw['load'],cfg))


def epoch_validation(windows, cfg, sensor, limits):
    """No averaging a held-out run's other endpoint into its predicted zero."""
    by_run = defaultdict(list)
    for w in windows:
        by_run[w['run']].append(w)
    means = {name: np.mean([w['bias_sensor_FLU'] for w in ws], axis=0)
             for name, ws in by_run.items()}
    if len(means) < limits.minimum_runs:
        return dict(accepted=False, reason='fewer_than_two_distinct_runs', runs=len(means))
    errors = []
    for name, ws in by_run.items():
        predicted = np.median([v for n, v in means.items() if n != name], axis=0)
        err = at_cg(np.array([w['bias_sensor_FLU'] for w in ws]) - predicted, cfg, sensor)
        errors.append(np.mean(err**2, axis=0))
    rmse = np.sqrt(np.mean(errors, axis=0))
    accepted = bool(np.all(rmse <= limits.wrench_scales))
    return dict(accepted=accepted, reason='passed' if accepted else 'cross_run_zero_disagreement',
                runs=len(means), endpoints=len(windows), leave_run_out_rmse=rmse.tolist(),
                bias_sensor_FLU=np.median(list(means.values()), axis=0).tolist())


def table(path, columns):
    return pd.read_csv(path, usecols=columns)


def locate_cache(run, roots):
    for root in roots:
        folder = root / run
        if (folder / 'force-grid/force_grid.csv').is_file():
            return folder
    raise ValueError('aligned_force_cache_unavailable')


def check_identity(run, folder, record):
    manifest = a.read_json(run / 'run.json')
    sensor = a.read_json(run / 'leptrino/summary.json')
    aligned = a.read_json(folder / 'aligned/summary.json')
    force = a.read_json(folder / 'force-grid/summary.json')
    boot = record.get('boot')
    digest = hashlib.sha256((run / 'run.json').read_bytes()).hexdigest()
    if not record.get('epoch') or not boot or sensor.get('zero_reset_sent'):
        raise ValueError('unknown_or_changing_sensor_epoch')
    ids = {v for v in (manifest.get('leptrino_boot_id'),
                       manifest.get('leptrino_boot_id_after'), sensor.get('boot_id')) if v}
    if ids != {boot}:
        raise ValueError('raw_sensor_boot_mismatch')
    for meta in (aligned, force):
        if meta.get('source_manifest_sha256') != digest or meta.get('leptrino_boot_id') != boot:
            raise ValueError('stale_or_wrong_clock_cache')
    if (aligned.get('common_clock') != 'aircraft CLOCK_MONOTONIC ns'
            or not aligned.get('aircraft_boot_id')
            or aligned.get('aircraft_boot_id') != force.get('aircraft_boot_id')):
        raise ValueError('clock_identity_mismatch')
    return aligned['aircraft_boot_id']


def gyro_data(run, control):
    px = control['px4_boot_ms'].to_numpy(float) * 1e6
    received = control['feedback_received_ns'].to_numpy(float)
    ok = np.isfinite(px) & np.isfinite(received) & (px > 0) & (received > 0)
    if ok.sum() < 20:
        raise ValueError('missing_px4_clock_pairs')
    origin = px[ok][0]
    slope, intercept = np.polyfit(px[ok] - origin, received[ok], 1)
    residual = received[ok] - (slope * (px[ok] - origin) + intercept)
    intercept += np.quantile(residual, .01) - slope * origin
    files = sorted((run / 'flightlog').glob('*.ulg'))
    if len(files) != 1:
        raise ValueError('ulog_not_unique')
    log = ULog(str(files[0]), message_name_filter_list=['vehicle_angular_velocity'])
    topics = [x.data for x in log.data_list if x.name == 'vehicle_angular_velocity']
    if len(topics) != 1:
        raise ValueError('gyro_topic_not_unique')
    d = topics[0]
    times = slope * np.asarray(d['timestamp_sample'], float) * 1000 + intercept
    values = np.column_stack([d[f'xyz[{j}]'] for j in range(3)])
    for drop in log.dropouts:
        end = slope * drop.timestamp * 1000 + intercept
        values[(times >= end - drop.duration * 1e6 - 50e6) & (times <= end + 50e6)] = np.nan
    return times, values


def extract_sides(run, folder, session, cfg, limits):
    """Read complete acquisition; frozen fit signals omit some stopped prefixes."""
    control = table(run / 'pid/control.csv', ['sent_ns', 'feedback_received_ns', 'px4_boot_ms']
                    + ['sent' + str(j) for j in range(1, 5)])
    u = control[[f'sent{j}' for j in range(1, 5)]].to_numpy(float)
    ct = control.sent_ns.to_numpy(np.int64)
    if not np.isfinite(u).all() or np.any(u < 0) or np.any(np.diff(ct) <= 0):
        raise ValueError('invalid_motor_command_log')
    active = ct[np.any(u > 0, axis=1)]
    if not len(active):
        raise ValueError('no_active_command_for_logger_health')
    edges = table(folder / 'aligned/rpm_edges_aligned.csv', ['channel', 'aircraft_estimate_ns']).dropna()
    if set(edges.channel) != {1, 2, 3, 4}:
        raise ValueError('rpm_health_not_observed_on_all_channels')
    first = min(active[0], edges.aircraft_estimate_ns.min())
    last = max(active[-1], edges.aircraft_estimate_ns.max())
    enc = table(folder / 'aligned/encoder_aligned.csv', ['aircraft_receive_estimate_ns',
                'experimental_pitch_deg', 'experimental_yaw_deg', 'within_clock_anchors'])
    enc = enc[enc.within_clock_anchors.astype(str).str.lower().isin(['true', '1'])].dropna()
    et = enc.aircraft_receive_estimate_ns.to_numpy(np.int64)
    angles = np.unwrap(np.deg2rad(enc[['experimental_pitch_deg', 'experimental_yaw_deg']]), axis=0)
    if np.any(np.diff(et) < 0):
        raise ValueError('encoder_clock_reversal')
    et, group, count = np.unique(et, return_inverse=True, return_counts=True)
    angles = np.column_stack([np.bincount(group, weights=angles[:, j]) / count for j in (0, 1)])
    meta = a.read_json(run / 'encoder/metadata.json')
    if (meta.get('pitch_sign') != cfg.pitch_encoder_sign
            or meta.get('offset_roll_deg') != cfg.pitch_encoder_offset_deg
            or meta.get('yaw_sign') != 1.0 or meta.get('offset_yaw_deg') != -83.1445):
        raise ValueError('encoder_calibration_mismatch')
    try:
        gt, gyro = gyro_data(run, control)
    except (ValueError,FileNotFoundError):
        gt,gyro=np.array([],float),np.empty((0,3))
    columns = ['estimated_measurement_aircraft_ns', 'receive_monotonic_ns', 'status',
               'fx_N', 'fy_N', 'fz_N', 'mx_Nm', 'my_Nm', 'mz_Nm']
    force = table(folder / 'force-grid/force_grid.csv', columns).dropna()
    ft = force.estimated_measurement_aircraft_ns.to_numpy(np.int64)
    if np.any(np.diff(ft) <= 0):
        raise ValueError('force_clock_reversal')
    sides = {}
    step = round(1e9 / cfg.hz)
    for side, low, high in [('pre', max(ft[0], et[0]), first-limits.motor_guard_s*1e9),
                            ('post', last+limits.motor_guard_s*1e9, min(ft[-1], et[-1]))]:
        if high-low < (min(limits.windows_s) + 1.)*1e9:
            sides[side] = dict(reason='insufficient_joint_sensor_coverage', available_s=float((high-low)/1e9))
            continue
        # At most 12 seconds adjacent to actuation; do not search long periods for low residuals.
        low, high = (max(low, high-12e9), high) if side == 'pre' else (low, min(high, low+12e9))
        grid = np.arange(int(np.ceil(low/step)*step), int(high), step, dtype=np.int64)
        ix = (ft >= grid[0]-step//2) & (ft < grid[-1]+step//2)
        bins = (ft[ix]-grid[0]+step//2)//step
        count = np.bincount(bins, minlength=len(grid))
        vals = force.iloc[np.flatnonzero(ix)][columns[3:]].to_numpy(float)
        load = np.column_stack([np.divide(np.bincount(bins, weights=vals[:, j], minlength=len(grid)),
                   count, out=np.full(len(grid), np.nan), where=count >= 6) for j in range(6)])
        status = np.zeros(len(grid), dtype=np.uint32)
        np.bitwise_or.at(status, bins, force.status.to_numpy(np.uint32)[ix])
        load[~a.leptrino_status_valid(status, accept_0x44=cfg.accept_leptrino_0x44)] = np.nan
        # Preserve USB-gap exclusion from the production loader.
        receipt = force.receive_monotonic_ns.to_numpy(np.int64)
        for k in np.flatnonzero(np.diff(receipt) > 30_000_000):
            end = ft[k+1]
            load[(grid >= end-(receipt[k+1]-receipt[k])) & (grid <= end)] = np.nan
        raw = dict(t=(grid-grid[0])/1e9, grid_origin_ns=int(grid[0]),
                   angles=a.interpolate(et, angles, grid, 30_000_000),
                   gyro=a.interpolate(gt, gyro, grid, 30_000_000), load=load,
                   heading=np.deg2rad(session.get('heading_deg', 0)), wind=0.)
        # Keep encoder-only windows when ULog starts with motor operation.
        fallback=encoder_motion(raw,cfg)
        a.prepare_motion(raw, cfg)
        fused=raw['valid_motion']
        for key in ('joint_angle','rotation','omega','alpha'):
            fallback[key][fused]=raw[key][fused]
        fallback['valid_motion'] |= fused
        fallback['motion_source']=np.where(fused,'encoder_gyro','encoder_only')
        sides[side] = fallback
    return sides


def choose_window(raw, cfg, sensor, limits):
    bias = motion_bias(raw, cfg, sensor)
    valid = raw['valid_motion'] & np.isfinite(bias).all(axis=1)
    failures = Counter()
    for duration in limits.windows_s:
        n = round(duration * cfg.hz)
        starts = range(0, len(valid)-n+1, cfg.hz//2)
        if raw['side'] == 'pre':
            starts = reversed(list(starts))
        for start in starts:
            ix = np.arange(start, start+n)
            if not valid[ix].all():
                failures['sensor_coverage_or_filter_edge'] += 1
                continue
            rate = float(np.max(np.linalg.norm(raw['omega'][ix], axis=1))*180/np.pi)
            accel = float(np.max(np.linalg.norm(raw['alpha'][ix], axis=1))*180/np.pi)
            if rate > limits.max_rate_deg_s or accel > limits.max_acceleration_deg_s2:
                failures['motion_too_fast_for_zero_aerodynamics_assumption'] += 1
                continue
            # Window selected using only timing/kinematics, not residual magnitude.
            b = bias[ix]
            mean = b.mean(axis=0)
            residual = at_cg(b-mean, cfg, sensor)
            blocks = np.array([b[k:k+cfg.hz].mean(axis=0) for k in range(0, n, cfg.hz)])
            shift = at_cg(b[n//2:].mean(axis=0)-b[:n//2].mean(axis=0), cfg, sensor)
            rms = np.sqrt(np.mean(residual**2, axis=0))
            passed = bool(np.all(rms <= limits.wrench_scales) and
                          np.all(abs(shift) <= limits.wrench_scales))
            angle = np.rad2deg(raw['joint_angle'][ix])
            static_raw = dict(raw, omega=np.zeros_like(raw['omega']), alpha=np.zeros_like(raw['alpha']))
            static_bias = motion_bias(static_raw, cfg, sensor)[ix].mean(axis=0)
            row = dict(window_seconds=duration, start_ns=int(raw['grid_origin_ns']+raw['t'][ix[0]]*1e9),
                       end_ns=int(raw['grid_origin_ns']+(raw['t'][ix[-1]]+1/cfg.hz)*1e9),
                       center_ns=float(raw['grid_origin_ns']+raw['t'][ix].mean()*1e9),
                       samples=n, bias_sensor_FLU=mean.tolist(), block_bias_sensor_FLU=blocks.tolist(),
                       residual_rms_CG=rms.tolist(), half_difference_CG=shift.tolist(),
                       motion_minus_gravity_only_CG=at_cg(mean-static_bias,cfg,sensor).tolist(),
                       max_rate_deg_s=rate, max_acceleration_deg_s2=accel,
                       motion_sources=dict(Counter(raw['motion_source'][ix])),
                       angle_std_deg=angle.std(axis=0).tolist(), angle_span_deg=np.ptp(angle,axis=0).tolist(),
                       window_consistent=passed, reason='passed' if passed else 'within_window_zero_disagreement',
                       reference_kind='motion_corrected_absolute',
                       limitation='Conditional on slow-motion negligible aerodynamics, no contact, and frozen mechanics')
            shorter=encoder_motion(raw,cfg,seconds=.5)
            sensitivity=motion_bias(shorter,cfg,sensor)[ix].mean(axis=0)-mean
            row['half_second_encoder_sensitivity_CG']=at_cg(sensitivity,cfg,sensor).tolist()
            return row, dict(time=raw['t'][ix], angle_deg=angle, bias_sensor_FLU=b,
                             load_sensor_FLU=raw['filtered_load'][ix],
                             expected_sensor_FLU=predict_sensor_load(raw,cfg,sensor)[ix])
    return dict(reason='no_eligible_window', rejection_counts=dict(failures)), None


def main(args):
    profile = a.read_json(args.bundle/'profile.json')
    cfg = a.Settings(**profile['settings'])
    sensor = a.sensor_parameters(profile['sensor_summary'])
    limits = Limits()
    records = a.read_json(args.records)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'windows').mkdir()
    old_keys = {(x['epoch'],x['aircraft_boot_id']) for x in profile['anchors']}
    args.aligned_root += sorted({Path(x['provenance']).parent for x in profile['anchors']})
    current_runs = {r['run']:r for r in profile['runs']}
    attempts, windows = [], []
    for record in records:
        run = args.raw_root/record['run']
        paths = [run/f for f in ['resolved-session.json','operator-session.json','source-session.json']]
        session_path = next((p for p in paths if p.exists()), None)
        if session_path is None:
            continue
        session = a.read_json(session_path)
        manifest = a.read_json(run/'run.json')
        wind, basis = a.reported_wind(session, manifest.get('wind_condition',{}))
        if wind != 0 or session.get('role') not in ('T','D','H'):
            continue
        report = dict(run=run.name, role=session['role'], wind=wind, wind_basis=basis,
                      epoch=record.get('epoch'), selected_for_aero_fit=run.name in current_runs)
        attempts.append(report)
        try:
            folder = locate_cache(run.name,args.aligned_root)
            boot = check_identity(run,folder,record)
            report.update(aircraft_boot_id=boot, old_anchored_epoch=(record['epoch'],boot) in old_keys)
            sides = extract_sides(run,folder,session,cfg,limits)
            report['sides'] = {}
            for side,raw in sides.items():
                if 'reason' in raw:
                    report['sides'][side] = raw
                    continue
                raw['side'] = side
                row, data = choose_window(raw,cfg,sensor,limits)
                report['sides'][side] = row
                if data is not None:
                    row.update(run=run.name, side=side, epoch=record['epoch'], role=session['role'],
                               wind=0., aircraft_boot_id=boot, provenance=str(folder))
                    windows.append(row)
                    np.savez_compressed(args.output/'windows'/f'{run.name}_{side}.npz',**data)
            print('AUDIT',run.name,session['role'],{k:v.get('reason') for k,v in report['sides'].items()},flush=True)
        except (ValueError,FileNotFoundError,KeyError) as ex:
            report['rejected_reason'] = str(ex)
            print('SKIP',run.name,str(ex),flush=True)
    by_key = defaultdict(list)
    for w in windows:
        if w['window_consistent']:
            by_key[w['epoch'],w['aircraft_boot_id']].append(w)
    epochs, accepted = [], []
    for key, ws in by_key.items():
        check = epoch_validation(ws,cfg,sensor,limits)
        members = [r['run'] for r in profile['runs'] if tuple(r['key'])==key]
        row = dict(key=list(key),old_anchored=key in old_keys,affected_runs=members,**check)
        epochs.append(row)
        if check['accepted'] and key not in old_keys:
            accepted.extend(ws)
    accepted_keys = {(w['epoch'],w['aircraft_boot_id']) for w in accepted}
    new_runs = [r['run'] for r in profile['runs'] if not r['anchored'] and tuple(r['key']) in accepted_keys]
    result = dict(schema='fftgyro-motion-zero-audit-v1', limits=asdict(limits),
                  profile_sha256=hashlib.sha256((args.bundle/'profile.json').read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  analysis_code_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
                  attempts=attempts,windows=windows,epochs=epochs,accepted_anchors=accepted,
                  newly_supported_runs=new_runs,
                  policy='T/D/H independent no-wind motor-off intervals; no aerodynamic-fit residual used',
                  limitations=['Moving-aircraft aerodynamic/contact loads are not independently measured.',
                    'Cross-run consistency checks bound repeatability, not shared calibration error.',
                    'Same epoch interpolation/endpoint holding retains unobserved drift uncertainty.',
                    'New references are motion-corrected conditional references, not certified absolute truth.',
                    'Existing static references and frozen model coefficients have not been changed.'])
    a.write_json(args.output/'summary.json',result)
    print('DONE',dict(attempts=len(attempts),windows=len(windows),new_runs=len(new_runs)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('bundle','records','raw-root','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--aligned-root',type=Path,action='append',required=True)
    main(p.parse_args())
