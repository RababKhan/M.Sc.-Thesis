# Insert the Paper-3 figures into the built manuscript with Word, save it, and export a PDF for checking.
#   powershell -File experiments/paper3/paper_word.ps1 <in.docx> <figure dir> <out.docx> <out.pdf>
param([string]$InDocx, [string]$FigDir, [string]$OutDocx, [string]$OutPdf)
$ErrorActionPreference = "Stop"
$figs = @{ 1 = "fig1_latency_vs_reduction.png"; 2 = "fig2_proxy_validation.png"; 3 = "fig3_resnet8_latency_room.png" }
Copy-Item $InDocx $OutDocx -Force
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($OutDocx)
    foreach ($n in 1..3) {
        $range = $doc.Content
        $find = $range.Find
        $find.ClearFormatting()
        if (-not $find.Execute("[[FIG$n]]")) { throw "marker [[FIG$n]] not found" }
        $range.Text = ""
        $shape = $doc.InlineShapes.AddPicture((Join-Path $FigDir $figs[$n]), $false, $true, $range)
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
