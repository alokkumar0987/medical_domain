@echo off
REM Activate venv myenv011 and run lung pipeline: report -> NLP -> TotalSegmentator
REM Run from project root: scripts\run_lung_pipeline_myenv011.bat
REM Or with CT path: scripts\run_lung_pipeline_myenv011.bat "D:\path\to\LIDC-IDRI-0004"

set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

if exist "myenv011\Scripts\activate.bat" (
    call myenv011\Scripts\activate.bat
) else if exist "myenv001\Scripts\activate.bat" (
    echo myenv011 not found, using myenv001
    call myenv001\Scripts\activate.bat
) else (
    echo No venv myenv011 or myenv001 found. Using current Python.
)

python scripts\run_lung_report_nlp_totalseg.py %*
exit /b %ERRORLEVEL%
