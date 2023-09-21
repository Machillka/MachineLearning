$trainPath = "EdgeLogoTraining.py"
$initPath = "EdgeLogoInit.py"
$epochs = 300

python.exe $initPath

Write-Host "Training Start!"

for ($i = 0; $i -le $epochs; $i++)
{
    python.exe $trainPath
    Write-Host "Training Successfully!  ", $i + 1, "Times"
}
