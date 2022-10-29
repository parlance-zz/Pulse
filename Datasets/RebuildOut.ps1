# this script will rebuild the input raw audio files from the quantized spike intervals
$errorActionPreference = "Stop"

# include common dataset parameters
. ./Common.ps1
$sourcePath = "Quants"
$destPath = "Out"
if ($args.count -eq 0) { $pathList = @("*") }
else { $pathList = $args }
for ($i = 0; $i -lt $pathList.count; $i++) { $pathList[$i] = $sourcePath + "/" + $pathList[$i] }

$sourceFiles = Get-ChildItem $pathList -Recurse
ForEach ($source in $sourceFiles)
{
	$qFile = $false
	if ($source.extension -like ".q") { $qFile = $true }
	if ($qFile -eq $false) { Continue }
	
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
		try { $existing_processes = get-process pulse } catch { break }
		if ($existing_processes.Count -ge $MAX_JOBS) { Start-Sleep -s 1 }
		else { break }
	}
	$errorActionPreference = "Stop"

	"& $pulsePath -d $inputPath $outputPath"
	Start-Process -NoNewWindow -WorkingDirectory "." -FilePath $pulsePath -ArgumentList "-d `"$($inputPath)`" `"$($outputPath)`""
}

if ($sourceFiles.count -eq 0) { Write-Host "No input files found." }