$tempPath = "Temp"
$ffmpegPath = "../FFMpeg/bin/ffmpeg.exe"
#$pulsePath = "../Pulse/x64/Debug/Pulse.exe"
$pulsePath = "../Pulse/x64/Release/Pulse.exe"

try # use half the number of available logical cores for dataset tasks
{
    $processor = Get-ComputerInfo -Property CsProcessors
    $MAX_JOBS = [int]($processor.CsProcessors.NumberOfCores/2)
}
catch
{
    $MAX_JOBS = 1
}