$sourcePath = "Source"
$destPath = "Raw"
$ffmpegPath = "..\FFMpeg\bin"

$tempPath = "Temp\temp" + ([string](Get-Random)) + ".raw"

$sourceFiles = Get-ChildItem ($sourcePath) -Recurse

ForEach ($source in $sourceFiles)
{
	$mediaFile = $false
	
	if ($source.extension -like ".mp3") { $mediaFile = $true }
	if ($source.extension -like ".flac") { $mediaFile = $true }
	
	if ($mediaFile -eq $false) { Continue }
	
	$inputPath = $source.FullName
	$outputPath = $destPath + "\" + $source.BaseName + ".raw"
	
	if (Test-Path $outputPath)
	{
		"Skipping " + $outputPath + "..."
		Continue
	}
	
	"$ffmpegPath\ffmpeg.exe -y -i $inputPath -af loudnorm=I=-16:LRA=11:TP=-1.5 -ac 1 -ar 48000 -f f32le $outputPath"
	& "$ffmpegPath\ffmpeg.exe" -y -i $inputPath -af loudnorm=I=-16:LRA=11:TP=-1.5 -ac 1 -ar 48000 -f f32le $outputPath
}
