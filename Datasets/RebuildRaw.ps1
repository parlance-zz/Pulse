# this script will rebuild single channel float32 48khz raw audio files from source mp3 and flac files
$errorActionPreference = "Stop"

# include common dataset parameters
. ./Common.ps1
$sourcePath = "Source"
$destPath = "Raw"
if ($args.count -eq 0) { $pathList = @("*") }
else { $pathList = $args }
for ($i = 0; $i -lt $pathList.count; $i++) { $pathList[$i] = $sourcePath + "/" + $pathList[$i] }

$sourceFiles = Get-ChildItem $pathList -Recurse
ForEach ($source in $sourceFiles)
{
	$mediaFile = $false
	if ($source.extension -like ".mp3") { $mediaFile = $true }
	if ($source.extension -like ".flac") { $mediaFile = $true }
	if ($mediaFile -eq $false) { Continue }
	
	$inputPath = $source.FullName
	$inputFolder = $source.Directory.BaseName
	if ($inputFolder -like $sourcePath) # file is in root $sourcePath
	{
		$outputPath = $destPath + "/" + $source.BaseName + ".raw"
	}
	else
	{
		$outputFolder = $destPath + "/" + $inputFolder # file is in subfolder of $sourcePath
		$outputPath = $outputFolder + "/" + $source.BaseName + ".raw"
		if ((Test-Path $outputFolder -PathType Container) -eq $false) # ensure output path exists
		{
			New-Item -ItemType Directory -Force -Path ($destPath + "/" + $inputFolder)
		}
	}

	if (Test-Path $outputPath) # skip completed files
	{
		"Skipping " + $outputPath + "..."
		Continue
	}

	# wait for a free job slot
	$errorActionPreference = "SilentlyContinue"
	while ($true)
	{
		try { $existing_processes = get-process ffmpeg } catch { break }
		if ($existing_processes.Count -ge $MAX_JOBS) { Start-Sleep -s 1 }
		else { break }
	}
	$errorActionPreference = "Stop"

	"& $ffmpegPath -y -i $inputPath -af loudnorm=I=-16:LRA=11:TP=-1.5 -ac 1 -ar 48000 -f f32le $outputPath"
	Start-Process -NoNewWindow -WorkingDirectory "." -FilePath $ffmpegPath -ArgumentList "-y -i `"$($inputPath)`" -af loudnorm=I=-16:LRA=11:TP=-1.5 -ac 1 -ar 48000 -f f32le  `"$($outputPath)`""
}

if ($sourceFiles.count -eq 0) { Write-Host "No input files found." }