$ErrorActionPreference = "Stop"

$Python = "D:\Anaconda3\envs\yolo\python.exe"
$Script = "D:\gogelila\Eating_behavior\yolo\train_pose.py"

if (-not (Test-Path $Python)) {
    throw "Missing conda yolo Python: $Python"
}

& $Python $Script @args
