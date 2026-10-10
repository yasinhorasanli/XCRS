// Print html/<id>.html to pdf/<id>.pdf with Chrome (see make_cvs.py). The layout comes from cases.yaml:
// LinkedIn-style exports get "Page n of m" footers, and the scanned case becomes an image-only PDF (no text).
import { chromium } from 'playwright-core'
import { mkdirSync, readFileSync, readdirSync } from 'node:fs'

const CHROME = process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const layouts = Object.fromEntries(
  [...readFileSync('cases.yaml', 'utf8').matchAll(/- id: (\S+)\n\s+layout: (\S+)/g)].map((m) => [m[1], m[2]]),
)
mkdirSync('pdf', { recursive: true })
const browser = await chromium.launch({ executablePath: CHROME })
const page = await browser.newPage({ viewport: { width: 794, height: 1123 } })
for (const file of readdirSync('html').filter((f) => f.endsWith('.html'))) {
  const id = file.replace(/\.html$/, '')
  await page.setContent(readFileSync(`html/${file}`, 'utf8'))
  if (layouts[id] === 'scanned') {
    const png = await page.screenshot({ fullPage: true, type: 'png' })
    await page.setContent(
      `<style>body{margin:0}img{width:100%}</style><img src="data:image/png;base64,${png.toString('base64')}">`,
    )
    await page.pdf({ path: `pdf/${id}.pdf`, format: 'A4' })
  } else {
    const footer = layouts[id] === 'linkedin'
    await page.pdf({
      path: `pdf/${id}.pdf`,
      format: 'A4',
      printBackground: true,
      displayHeaderFooter: footer,
      headerTemplate: '<span></span>',
      footerTemplate: footer
        ? '<div style="font-size:8px;width:100%;text-align:right;margin-right:34px;color:#888">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>'
        : '<span></span>',
      margin: footer ? { top: '20px', bottom: '40px' } : undefined,
    })
  }
  console.log(`pdf/${id}.pdf`)
}
await browser.close()
