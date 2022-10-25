$sourcePath = "Quants"
$destPath = "Out"
$pulsePath = "..\Pulse\x64\Release"

$sourceFiles = Get-ChildItem ($sourcePath)

ForEach ($source in $sourceFiles)
{
	$qFile = $false
	if ($source.extension -like ".q") { $qFile = $true }
	if ($qFile -eq $false) { Continue }
	
	$inputPath = $source.FullName
	$outputPath = $inputPath.Replace($source.extension, ".raw").Replace($sourcePath, "").Replace("\", "")
	$outputPath = $destPath + "\" + $outputPath
	
	& "$pulsePath\pulse.exe" -d $inputPath $outputPath
}
