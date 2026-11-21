param(
    [ValidatePattern('^COM\d+$')]
    [string]$Port = 'COM7',
    [ValidateRange(1, 4)]
    [int]$Motor = 1,
    [ValidateRange(1000, 1200)]
    [int]$Pwm = 1100,
    [ValidateRange(1, 10)]
    [int]$DurationSeconds = 5
)

$ErrorActionPreference = 'Stop'

function Update-X25Crc {
    param([int]$Crc, [int]$Value)
    $tmp = ($Value -bxor ($Crc -band 0xff)) -band 0xff
    $tmp = ($tmp -bxor (($tmp -shl 4) -band 0xff)) -band 0xff
    $next = ($Crc -shr 8) -bxor (($tmp -shl 8) -band 0xffff)
    $next = $next -bxor (($tmp -shl 3) -band 0xffff)
    $next = $next -bxor ($tmp -shr 4)
    return $next -band 0xffff
}

$serial = [System.IO.Ports.SerialPort]::new($Port, 115200, 'None', 8, 'One')
$serial.ReadTimeout = 200
$serial.WriteTimeout = 1000
$serial.DtrEnable = $false
$serial.RtsEnable = $false

try {
    $serial.Open()

    $stream = [System.IO.MemoryStream]::new()
    $writer = [System.IO.BinaryWriter]::new($stream)
    $writer.Write([single]$Motor)
    $writer.Write([single]1)
    $writer.Write([single]$Pwm)
    $writer.Write([single]$DurationSeconds)
    $writer.Write([single]0)
    $writer.Write([single]0)
    $writer.Write([single]0)
    $writer.Write([uint16]209)
    $writer.Write([byte]1)
    $writer.Write([byte]1)
    $writer.Write([byte]0)
    $payload = $stream.ToArray()

    $header = [byte[]](0xfe, 33, 1, 255, 190, 76)
    $crc = 0xffff
    foreach ($value in $header[1..5]) {
        $crc = Update-X25Crc $crc $value
    }
    foreach ($value in $payload) {
        $crc = Update-X25Crc $crc $value
    }
    $crc = Update-X25Crc $crc 152

    $packet = [byte[]]::new(41)
    [Array]::Copy($header, 0, $packet, 0, 6)
    [Array]::Copy($payload, 0, $packet, 6, 33)
    $packet[39] = [byte]($crc -band 0xff)
    $packet[40] = [byte](($crc -shr 8) -band 0xff)
    $serial.Write($packet, 0, $packet.Length)
    Write-Host "M${Motor}: ${Pwm} us for ${DurationSeconds} seconds"
}
finally {
    if ($serial.IsOpen) {
        $serial.Close()
    }
    $serial.Dispose()
}
