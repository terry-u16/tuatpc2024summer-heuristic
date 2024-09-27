Write-Host "[Compile]"
cargo build --release
Move-Item ../target/release/tuatpc2024summer-heuristic.exe . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "1.5"
dotnet marathon run-local