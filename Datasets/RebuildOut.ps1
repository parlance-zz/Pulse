$errorActionPreference = "Stop"

# include common dataset parameters
. ./Common.ps1
$sourcePath = "Quants"
$destPath = "Out"

$sourceFiles = Get-ChildItem ($sourcePath) -Recurse
ForEach ($source in $sourceFiles)
{
	$qFile = $false
	if ($source.extension -like ".q") { $qFile = $true }
	if ($qFile -eq $false) { Continue }
	
	$inputPath = $source.FullName
	$outputPath = $destPath + "\" + $source.BaseName + ".raw"
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

	"& $pulsePath\pulse.exe -d $inputPath $outputPath"
	Start-Process -NoNewWindow -WorkingDirectory "." -FilePath ("$pulsePath\pulse.exe") -ArgumentList "-d `"$($inputPath)`" `"$($outputPath)`""
}
