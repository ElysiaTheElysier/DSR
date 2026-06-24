$logFile = "C:\Ki_5\final\ev_car\shutdown_watcher.log"
"Starting shutdown watcher..." | Out-File $logFile
while ($true) {
    $procs = Get-WmiObject Win32_Process -Filter "CommandLine LIKE '%llm_openai_extraction.py%'" | Where-Object { $_.Name -match "python.exe" }
    if (-not $procs) {
        "Extraction process finished! Triggering shutdown." | Out-File $logFile -Append
        shutdown /s /t 60 /c "Data extraction complete. Shutting down in 60 seconds. Use 'shutdown /a' in run prompt to cancel."
        break
    }
    Start-Sleep -Seconds 30
}
