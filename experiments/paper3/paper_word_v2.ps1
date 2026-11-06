# Insert the Paper-3 figures into the built manuscript with Word, save it, and export a PDF for checking.
#   powershell -File experiments/paper3/paper_word_v2.ps1 <in.docx> <out.docx> <out.pdf> <fig1.png>,<fig2.png>,...
# Figure n replaces the marker [[FIGn]]; every figure is set to the 240 pt column width.
param([string]$InDocx, [string]$OutDocx, [string]$OutPdf, [string[]]$Figures)
$ErrorActionPreference = "Stop"
$Figures = @($Figures | ForEach-Object { $_ -split "," })     # -File passes the list as one comma-joined string
Copy-Item $InDocx $OutDocx -Force
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($OutDocx)
    for ($n = 1; $n -le $Figures.Count; $n++) {
        $range = $doc.Content
        $find = $range.Find
        $find.ClearFormatting()
        if (-not $find.Execute("[[FIG$n]]")) { throw "marker [[FIG$n]] not found" }
        $range.Text = ""
        $shape = $doc.InlineShapes.AddPicture($Figures[$n - 1], $false, $true, $range)
        $shape.LockAspectRatio = -1
        $shape.Width = 240
        "figure $n inserted: {0:N1} x {1:N1} pt" -f $shape.Width, $shape.Height
    }
    $doc.Save()
    $doc.ExportAsFixedFormat($OutPdf, 17)
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close($false)
} finally {
    $word.Quit()
}
