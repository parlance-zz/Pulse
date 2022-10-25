$errorActionPreference = "Stop"

$sourcePath = "Raw"
$destPath = "Quants"
$pulsePath = "..\Pulse\x64\Release"
$MAX_JOBS = 16

$sourceFiles = Get-ChildItem ($sourcePath) -Recurse

ForEach ($source in $sourceFiles)
{
	$rawFile = $false
	if ($source.extension -like ".raw") { $rawFile = $true }
	if ($rawFile -eq $false) { Continue }
	
	$inputPath = $source.FullName
	$outputPath = $destPath + "\" + $source.BaseName + ".q"
	if (Test-Path $outputPath)
	{
		"Skipping " + $outputPath + "..."
		Continue
	}

	$errorActionPreference = "SilentlyContinue"
	while ($true)
	{
		try { $existing_processes = get-process pulse } catch { break }
		if ($existing_processes.Count -ge $MAX_JOBS) { Start-Sleep -s 1 }
		else { break }
	}
	$errorActionPreference = "Stop"

	"& $pulsePath\pulse.exe -q $inputPath $outputPath"
	Start-Process -NoNewWindow -WorkingDirectory "." -FilePath ("$pulsePath\pulse.exe") -ArgumentList "-q `"$($inputPath)`" `"$($outputPath)`""
}
