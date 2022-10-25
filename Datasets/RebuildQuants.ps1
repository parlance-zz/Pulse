$sourcePath = "Raw"
$destPath = "Quants"
$pulsePath = "..\Pulse\x64\Release"

$sourceFiles = Get-ChildItem ($sourcePath)

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
	
	& "$pulsePath\pulse.exe" -q $inputPath $outputPath
}
