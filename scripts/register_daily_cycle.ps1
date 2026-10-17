<#
.SYNOPSIS
    Registra (o elimina) las dos tareas programadas del ciclo diario.

.DESCRIPTION
    Crea en el Programador de tareas de Windows, para el usuario actual:

      SportsPredictor - Picks      Todos los dias a las 09:00.
                                   scripts\daily_cycle.py picks
                                   Liquida los picks pendientes y registra
                                   los nuevos de NFL y MLB.

      SportsPredictor - Cierre     Todos los dias de 10:00 a 23:00, cada
                                   30 minutos.
                                   scripts\daily_cycle.py closing
                                   Captura el CLV de los picks cuyo partido
                                   empieza en los proximos 45 minutos.
                                   Sin picks por cerrar no gasta creditos.

    Las horas son las del reloj de este equipo (Ciudad de Mexico, UTC-6).
    La fase de cierre no depende de horarios fijos: consulta gratis a que
    hora empieza cada partido, asi que el cambio de horario de EE. UU. no
    la afecta.

    Las tareas corren con pythonw.exe (sin ventana) y SOLO con la sesion
    del usuario iniciada: no se guarda ninguna contrasena.

    WakeToRun: si el equipo esta SUSPENDIDO, Windows lo despierta para
    correr la tarea. Requiere que el plan de energia permita los
    temporizadores de activacion (en esta PC: si con corriente, no con
    bateria). Apagado del todo, la ejecucion se pierde; la de picks se
    recupera al encender (StartWhenAvailable) y la de cierre no hace
    falta recuperarla porque la siguiente corrida retoma lo pendiente.

    Registro: logs\daily_cycle\AAAA-MM-DD.log

.PARAMETER Unregister
    Elimina las dos tareas en vez de crearlas.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_cycle.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_cycle.ps1 -Unregister
#>

param(
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

$TaskPicks   = "SportsPredictor - Picks"
$TaskClosing = "SportsPredictor - Cierre"

if ($Unregister) {
    foreach ($name in @($TaskPicks, $TaskClosing)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Eliminada: $name"
        }
    }
    return
}

$Repo    = Split-Path -Parent $PSScriptRoot
$Python  = (Get-Command python.exe -ErrorAction Stop).Source
$Pythonw = Join-Path (Split-Path $Python) "pythonw.exe"
if (-not (Test-Path $Pythonw)) { $Pythonw = $Python }

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -MultipleInstances IgnoreNew

# ── Picks: una vez al dia ──────────────────────────────────────────────
$picksAction  = New-ScheduledTaskAction -Execute $Pythonw `
    -Argument "scripts\daily_cycle.py picks" -WorkingDirectory $Repo
$picksTrigger = New-ScheduledTaskTrigger -Daily -At "09:00"
$picksSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskPicks -Action $picksAction `
    -Trigger $picksTrigger -Settings $picksSettings -Force `
    -Description "Liquida y registra picks de NFL y MLB (scripts\daily_cycle.py picks)." | Out-Null
Write-Host "Registrada: $TaskPicks  (diaria 09:00)"

# ── Cierre: cada 30 minutos de 10:00 a 23:00 ───────────────────────────
$closingAction  = New-ScheduledTaskAction -Execute $Pythonw `
    -Argument "scripts\daily_cycle.py closing" -WorkingDirectory $Repo
$closingTrigger = New-ScheduledTaskTrigger -Daily -At "10:00"
$closingTrigger.Repetition = (New-ScheduledTaskTrigger -Once -At "10:00" `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Hours 13)).Repetition

Register-ScheduledTask -TaskName $TaskClosing -Action $closingAction `
    -Trigger $closingTrigger -Settings $settings -Force `
    -Description "Captura el CLV de cierre de los picks pendientes (scripts\daily_cycle.py closing)." | Out-Null
Write-Host "Registrada: $TaskClosing  (10:00-23:00 cada 30 min)"

Write-Host ""
Write-Host "Python : $Pythonw"
Write-Host "Carpeta: $Repo"
Write-Host "Log    : $Repo\logs\daily_cycle\"
