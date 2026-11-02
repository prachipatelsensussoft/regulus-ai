# Load a KEY=VALUE profile (configs/ui.env, configs/llm_model.env) into this PowerShell session's environment, as
# `set -a; source <profile>; set +a` does for the zsh launchers. Blank lines and # comments are skipped; one pair of
# surrounding quotes is removed from a value.
param([Parameter(Mandatory = $true)][string]$Path)
foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
    $text = $line.Trim()
    if ($text -eq "" -or $text.StartsWith("#")) { continue }
    $eq = $text.IndexOf("=")
    if ($eq -lt 1) { continue }
    $key = $text.Substring(0, $eq).Trim()
    $value = $text.Substring($eq + 1).Trim()
    if ($value.Length -ge 2 -and (($value[0] -eq '"' -and $value[-1] -eq '"') -or ($value[0] -eq "'" -and $value[-1] -eq "'"))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    Set-Item -Path "Env:$key" -Value $value
}
