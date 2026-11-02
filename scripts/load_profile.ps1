# Load a KEY=VALUE profile (configs/ui.env, configs/llm_model.env) into this PowerShell session's environment, as
# `set -a; source <profile>; set +a` does for the zsh launchers, with the same result: blank lines, # comments and a
# trailing " # comment" are skipped, and a quoted value is taken up to its closing quote.
param([Parameter(Mandatory = $true)][string]$Path)
foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
    $text = $line.Trim()
    if ($text -eq "" -or $text.StartsWith("#")) { continue }
    $eq = $text.IndexOf("=")
    if ($eq -lt 1) { continue }
    $key = $text.Substring(0, $eq).Trim()
    $value = $text.Substring($eq + 1).Trim()
    if ($value.Length -ge 1 -and ($value[0] -eq '"' -or $value[0] -eq "'")) {
        # a quoted value runs to its closing quote; a comment after it is dropped
        $end = $value.IndexOf($value[0], 1)
        if ($end -gt 0) { $value = $value.Substring(1, $end - 1) }
    } else {
        # as zsh reads it: whitespace then # starts a comment (KEY=auto   # note -> auto); a # inside a word stays
        $value = $value -replace '\s+#.*$', ''
    }
    Set-Item -Path "Env:$key" -Value $value
}
