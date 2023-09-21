@echo off
SET "StartPath=%cd%"
SET "ToPath=s1620409@pcc:/home/s1620409/WORK/SimilarityCategoricalData"
SetLocal EnableDelayedExpansion

SET "scpstr="
FOR /R %%i IN (*.py) DO (
	set "SubDirsAndFiles=%%~fi"
	set "SubDirsAndFiles=!SubDirsAndFiles:%StartPath%=!" 
	set "ToFile=!ToPath!!SubDirsAndFiles!"
	REM #ECHO !ToFile! 
	scp .!SubDirsAndFiles! !ToFile! 
	REM SET "scpstr=!scpstr! .!SubDirsAndFiles!"
)

REM ECHO !scpstr!
REM scp !scpstr! !ToPath!
