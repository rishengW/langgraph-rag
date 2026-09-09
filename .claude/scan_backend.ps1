# Diffs src/backend against a persisted snapshot and reports only deltas.
# Steady state (no changes) costs one directory walk: unchanged files are accepted on
# len+mtime and keep their stored SHA, so file contents are never re-read. Content
# hashes are computed only for added files and for files whose len or mtime moved.
# Usage: powershell -File .claude\scan_backend.ps1
param([string]$Root = 'E:\langgraph-rag', [string]$Target = 'src\backend')

$ErrorActionPreference = 'Stop'
$Snap   = Join-Path $Root '.claude\backend_snapshot.json'
$SnapDir = Split-Path $Snap -Parent
if (-not (Test-Path $SnapDir)) { New-Item -ItemType Directory -Force $SnapDir | Out-Null }

# Cheap pass: attributes only, never touches file contents.
function Get-Cheap([string]$path) {
    $h = @{}
    foreach ($f in (Get-ChildItem $path -Recurse -Force -File)) {
        $h[$f.FullName] = [ordered]@{ len = $f.Length; mtime = $f.LastWriteTime.ToString('s') }
    }
    return $h
}

# Full pass: adds SHA-256.
function Get-Full([string]$path) {
    $h = @{}
    foreach ($f in (Get-ChildItem $path -Recurse -Force -File)) {
        $h[$f.FullName] = [ordered]@{
            len   = $f.Length
            mtime = $f.LastWriteTime.ToString('s')
            sha   = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
        }
    }
    return $h
}

$rootPath   = Resolve-Path $Root
$targetPath = Join-Path $rootPath.Path $Target
$curCheap   = Get-Cheap $targetPath
$curCount   = $curCheap.Keys.Count

# PS 5.1 cannot build a generic HashSet[string] via New-Object(...($keys, comparer));
# it yields a set that rejects every Contains() check. Hashtables are case-insensitive by default.
function KeySet($keys) { $s = @{}; foreach ($k in $keys) { $s[$k] = $true }; return $s }

$prev = $null
if (Test-Path $Snap) {
    $prev = @{}
    $json = Get-Content $Snap -Raw | ConvertFrom-Json
    foreach ($prop in $json.PSObject.Properties) {
        $prev[$prop.Name] = [ordered]@{
            len   = $prop.Value.len
            mtime = $prop.Value.mtime
            sha   = $prop.Value.sha
        }
    }
}

$prevKeys = if ($null -eq $prev) { @() } else { @($prev.Keys) }
$curKeys  = @($curCheap.Keys)
$prevSet  = KeySet $prevKeys
$curSet   = KeySet $curKeys
$added    = @($curKeys | Where-Object { $null -eq $prev -or -not $prevSet.ContainsKey($_) } | Sort-Object)
$removed  = @()
if ($null -ne $prev) {
    $removed = @($prevKeys | Where-Object { -not $curSet.ContainsKey($_) } | Sort-Object)
}

"SCAN $(Get-Date -Format 'HH:mm:ss')  files={0}" -f $curCount

if ($null -eq $prev) {
    'FIRST RUN - baseline recorded, no diff'
    $cur | ConvertTo-Json -Depth 3 | Set-Content $Snap -Encoding utf8 -NoNewline
    "snapshot -> {0} ({1} entries)" -f $Snap, $curCount
    return
}

# Fast path: nothing added or removed and every remaining file is byte-identical in len+mtime.
$cheapChanged = @($curKeys | Where-Object {
    $p = $prev[$_]
    if ($null -eq $p) { return $false }
    $p.len -ne $curCheap[$_].len -or $p.mtime -ne $curCheap[$_].mtime
})
if (-not $added.Count -and -not $removed.Count -and -not $cheapChanged.Count) {
    'NO CHANGES since last scan'
    return
}

# Slow path: hash only what actually moved; others inherit the stored hash.
$full      = Get-Full $targetPath
$changedSet = KeySet $cheapChanged
foreach ($a in $added) { $changedSet[$a] = $true }
$changed = @(($cheapChanged + $added | Sort-Object) | Where-Object { $prevSet.ContainsKey($_) })

if ($added.Count) {
    '--- ADDED ({0}) ---' -f $added.Count
    foreach ($a in $added) { '{0,8:N0}  {1}' -f $curCheap[$a].len, ($a.Replace($rootPath.Path + '\', '')) }
}
if ($removed.Count) {
    '--- REMOVED ({0}) ---' -f $removed.Count
    foreach ($r in $removed) { '  {0}' -f ($r.Replace($rootPath.Path + '\', '')) }
}
if ($changed.Count) {
    '--- CHANGED ({0}) ---' -f $changed.Count
    foreach ($c in $changed) {
        $oldSha = $prev[$c].sha
        $newSha = if ($changedSet.ContainsKey($c)) { $full[$c].sha } else { $oldSha }
        $tag = if ($oldSha -ne $newSha) { 'CONTENT' } else { 'mtime-only' }
        '{0,8:N0}  {1}  {2}' -f $curCheap[$c].len, $tag, ($c.Replace($rootPath.Path + '\', ''))
    }
}

$snapOut = @{}
foreach ($k in $curKeys) {
    $snapOut[$k] = if ($changedSet.ContainsKey($k)) { $full[$k] }
                   else { [ordered]@{ len = $curCheap[$k].len; mtime = $curCheap[$k].mtime; sha = $prev[$k].sha } }
}
$snapOut | ConvertTo-Json -Depth 3 | Set-Content $Snap -Encoding utf8 -NoNewline
"snapshot -> {0} ({1} entries)" -f $Snap, $curCount
