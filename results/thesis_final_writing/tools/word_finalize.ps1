# Update the TOC / List of Figures / List of Tables and page fields of final_thesis_MIST.docx, label appendix page
# numbers as A-1, B-2, ... in those lists (MIST §3.3: appendices are paginated A-1, A-2, ...), save the document and
# export final_thesis_MIST.pdf.
#   powershell -NoProfile -ExecutionPolicy Bypass -File results/thesis_final_writing/tools/word_finalize.ps1
# Note: the appendix prefixes are written into the generated list text after the last field update; if the fields are
# updated again in Word, run this script again (or the prefixes disappear from the lists, not from the pages).
param([string]$Dir = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)))
$ErrorActionPreference = "Stop"
$docx = Join-Path $Dir "final_thesis_MIST.docx"
$pdf = Join-Path $Dir "final_thesis_MIST.pdf"
function Log($m) { Write-Output ("{0:HH:mm:ss} {1}" -f (Get-Date), $m) }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    Log "open"
    $d = $word.Documents.Open($docx, $false, $false)
    # Built-in TOC styles (wdStyleTOC1 = -20 ... TOC5 = -24): MIST TP.14 (single, 3 pt; indents 0/6/18/31 mm from the
    # margin) and TP.15/16 (LoF/LoT: single, 6 pt, description at 31 mm from the margin).
    $mm = 2.8346
    $ind = @{ 1 = 0; 2 = 6; 3 = 18; 4 = 31 }
    for ($lv = 1; $lv -le 5; $lv++) {
        $s = $d.Styles.Item(-19 - $lv)
        $s.Font.Name = "Times New Roman"; $s.Font.Size = 12; $s.Font.Bold = $false
        $f = $s.ParagraphFormat
        $f.LineSpacingRule = 0; $f.Alignment = 0
        [void]$f.TabStops.ClearAll()
        if ($lv -le 4) {
            $f.SpaceBefore = 3; $f.SpaceAfter = 3; $f.LeftIndent = $ind[$lv] * $mm; $f.FirstLineIndent = 0
        } else {
            $f.SpaceBefore = 6; $f.SpaceAfter = 6; $f.LeftIndent = 31 * $mm; $f.FirstLineIndent = -31 * $mm
            [void]$f.TabStops.Add(31 * $mm, 0, 0)
        }
        [void]$f.TabStops.Add(153 * $mm, 2, 1)
    }
    Log "TOC styles set"
    for ($pass = 1; $pass -le 2; $pass++) {
        $k = 0
        foreach ($t in $d.TablesOfContents) { $k++; $t.Update(); Log "pass ${pass} - list ${k} updated" }
    }
    $fixed = 0
    foreach ($t in $d.TablesOfContents) {
        foreach ($p in $t.Range.Paragraphs) {
            $txt = $p.Range.Text
            $letter = $null
            if ($txt -match '^APPENDIX ([A-Z])\b') { $letter = $Matches[1] }
            elseif ($txt -match '^Table ([A-Z])\.\d+') { $letter = $Matches[1] }
            if ($letter) {
                while ($p.Range.Fields.Count -gt 0) { $p.Range.Fields.Item(1).Unlink() }
                $p.Range.Font.Reset()
                $p.Range.Font.Underline = 0
                $p.Range.Font.Color = -16777216
                $end = $p.Range.End - 1
                $r = $d.Range($end, $end)
                [void]$r.MoveStartWhile("0123456789", -20)
                if ($r.Text -match '^\d+$') { $r.InsertBefore("$letter-"); $fixed++ }
            }
        }
    }
    Log "appendix page labels fixed: $fixed"
    $n = $d.ComputeStatistics(2)
    Log "pages=$n"
    $d.Save()
    Log "saved"
    $d.ExportAsFixedFormat([string]$pdf, 17)
    Log "pdf exported"
    $d.Close($false)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    Log "done"
}
