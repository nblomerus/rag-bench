import React, { useState, useRef, useEffect } from 'react'
import { XIcon, EyeIcon } from './Icons'
import { API_BASE } from '../utils/api'

export function PaperViewer({ paper, highlightChunkId, highlightText, onClose, isEmpty }) {
    const [pdfError, setPdfError] = useState(false)
    const [pdfDoc, setPdfDoc] = useState(null)
    const [currentPage, setCurrentPage] = useState(1)
    const [totalPages, setTotalPages] = useState(0)
    const [isSearching, setIsSearching] = useState(false)
    const [searchStatus, setSearchStatus] = useState('')
    const [highlightBoxes, setHighlightBoxes] = useState([])
    const [renderScale, setRenderScale] = useState(1.5)
    const canvasRef = useRef(null)
    const containerRef = useRef(null)

    // Close on Escape key
    useEffect(() => {
        const handler = (e) => { if (e.key === 'Escape') onClose() }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [onClose])

    // Find the cited chunk for the floating reference card
    let citedChunk = null
    if (paper && highlightChunkId) {
        citedChunk = paper.chunks.find(c => c.chunk_id === highlightChunkId)
    }
    if (!citedChunk && paper && highlightText) {
        const preview = highlightText.slice(0, 100).toLowerCase()
        citedChunk = paper.chunks.find(c => c.text && c.text.toLowerCase().includes(preview))
    }

    const hasPdf = paper && !!paper.arxiv_id
    const pdfUrl = hasPdf ? `${API_BASE}/papers/${encodeURIComponent(paper.paper_id || paper.arxiv_id)}/pdf` : ''

    // Load PDF document
    useEffect(() => {
        if (!pdfUrl) return

        let cancelled = false

        async function loadPDF() {
            try {
                // Use global pdfjsLib from CDN (kept in index.html for worker compatibility)
                if (!window.pdfjsLib) {
                    console.error('PDF.js library not loaded')
                    setPdfError(true)
                    return
                }

                window.pdfjsLib.GlobalWorkerOptions.workerSrc =
                    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js'

                const loadingTask = window.pdfjsLib.getDocument(pdfUrl)
                const pdf = await loadingTask.promise

                if (cancelled) return
                setPdfDoc(pdf)
                setTotalPages(pdf.numPages)
                setPdfError(false)
            } catch (error) {
                console.error('Error loading PDF:', error)
                if (!cancelled) setPdfError(true)
            }
        }

        loadPDF()
        return () => { cancelled = true }
    }, [pdfUrl])

    // Render current page
    useEffect(() => {
        if (!pdfDoc || !canvasRef.current || currentPage < 1 || currentPage > totalPages) return

        let cancelled = false

        async function renderPage() {
            try {
                const page = await pdfDoc.getPage(currentPage)
                if (cancelled) return

                const canvas = canvasRef.current
                const context = canvas.getContext('2d')

                const RENDER_SCALE = 1.5
                const viewport = page.getViewport({ scale: RENDER_SCALE })

                canvas.width = viewport.width
                canvas.height = viewport.height

                const renderContext = {
                    canvasContext: context,
                    viewport: viewport
                }

                await page.render(renderContext).promise
                setRenderScale(RENDER_SCALE)
            } catch (error) {
                console.error('Error rendering page:', error)
            }
        }

        renderPage()
        return () => { cancelled = true }
    }, [pdfDoc, currentPage, totalPages])

    // Search for cited text and navigate to it
    useEffect(() => {
        if (!pdfDoc || !citedChunk || !citedChunk.text) return

        let cancelled = false

        async function searchAndHighlight() {
            setIsSearching(true)
            setSearchStatus('Searching...')
            setHighlightBoxes([])

            try {
                // Strip contextual prefix added during chunking: "Title — Section\n\n"
                let chunkText = citedChunk.text.trim()
                const prefixMatch = chunkText.match(/^.+ — .+\n\n/)
                if (prefixMatch) {
                    chunkText = chunkText.slice(prefixMatch[0].length).trim()
                }

                // Skip leading section headers and figure/table captions
                const lines = chunkText.split('\n')
                let contentStart = 0
                let inCaption = false
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i].trim()
                    if (!line) { inCaption = false; contentStart = i + 1; continue }
                    if (/^(Figure|Fig\.|Table|Algorithm)\s+\d+/i.test(line)) {
                        inCaption = true; contentStart = i + 1; continue
                    }
                    if (inCaption && line.length < 50 && /^[a-z]/.test(line)) {
                        contentStart = i + 1; continue
                    }
                    inCaption = false
                    if (line.length < 60 && !/[.!?:,]$/.test(line)) {
                        contentStart = i + 1; continue
                    }
                    break
                }
                if (contentStart > 0 && contentStart < lines.length) {
                    chunkText = lines.slice(contentStart).join('\n').trim()
                }

                const firstWords = chunkText.split(/\s+/).slice(0, 8).join(' ').toLowerCase()
                const searchWords = chunkText.slice(0, 150).toLowerCase()
                    .split(/\s+/).filter(w => w.length > 2)

                function findPassageStart(items) {
                    for (let i = 0; i < items.length; i++) {
                        let concat = ''
                        for (let j = i; j < Math.min(i + 15, items.length); j++) {
                            concat += (j > i ? ' ' : '') + items[j].str.trim()
                            const concatLower = concat.toLowerCase()
                            if (concatLower.length >= firstWords.length && concatLower.startsWith(firstWords)) {
                                return items[i]
                            }
                        }
                    }
                    return null
                }

                function makeHighlight(item, page) {
                    const RENDER_SCALE = 1.5
                    const viewport = page.getViewport({ scale: RENDER_SCALE })
                    const x0 = item.transform[4]
                    const y0 = item.transform[5]
                    const w = item.width * RENDER_SCALE
                    const h = (item.height || item.transform[3]) * RENDER_SCALE
                    const x = x0 * RENDER_SCALE
                    const canvasY = viewport.height - (y0 * RENDER_SCALE)
                    return {
                        leftPct: Math.max(0, (x / viewport.width) * 100),
                        topPct: Math.max(0, ((canvasY - h) / viewport.height) * 100),
                        widthPct: Math.max(0.5, (w / viewport.width) * 100),
                        heightPct: Math.max(0.5, (h / viewport.height) * 100),
                        isFirst: true
                    }
                }

                // Pass 1: strict prefix match
                let found = false
                for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
                    if (cancelled) return
                    const page = await pdfDoc.getPage(pageNum)
                    const textContent = await page.getTextContent()
                    const items = textContent.items.filter(it => it.str && it.str.trim())

                    const foundItem = findPassageStart(items)
                    if (foundItem && !cancelled) {
                        setCurrentPage(pageNum)
                        setHighlightBoxes([makeHighlight(foundItem, page)])
                        setSearchStatus(`Found on page ${pageNum}`)
                        found = true
                        break
                    }
                }

                // Pass 2: looser 3-word phrase match
                if (!found && !cancelled) {
                    const phrase = chunkText.split(/\s+/).slice(0, 3).join(' ').toLowerCase()
                    for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
                        if (cancelled) return
                        const page = await pdfDoc.getPage(pageNum)
                        const textContent = await page.getTextContent()
                        const items = textContent.items.filter(it => it.str && it.str.trim())
                        const pageText = items.map(it => it.str).join(' ').toLowerCase()

                        const matchCount = searchWords.filter(word => pageText.includes(word)).length
                        const requiredMatches = Math.max(2, Math.ceil(searchWords.length * 0.4))
                        if (matchCount < requiredMatches) continue

                        if (pageText.includes(phrase)) {
                            const phraseItem = items.find(it => it.str.toLowerCase().includes(phrase))
                            if (phraseItem && !cancelled) {
                                setCurrentPage(pageNum)
                                setHighlightBoxes([makeHighlight(phraseItem, page)])
                                setSearchStatus(`Found on page ${pageNum}`)
                                found = true
                                break
                            }
                        }
                    }
                }

                // Pass 3: last resort — navigate to page with most word overlap
                if (!found && !cancelled) {
                    let bestPage = 1
                    let bestCount = 0
                    for (let pageNum = 1; pageNum <= pdfDoc.numPages; pageNum++) {
                        if (cancelled) return
                        const page = await pdfDoc.getPage(pageNum)
                        const textContent = await page.getTextContent()
                        const pageText = textContent.items.map(it => it.str).join(' ').toLowerCase()
                        const matchCount = searchWords.filter(word => pageText.includes(word)).length
                        if (matchCount > bestCount) {
                            bestCount = matchCount
                            bestPage = pageNum
                        }
                    }
                    setCurrentPage(bestPage)
                    setSearchStatus(`Navigated to page ${bestPage}`)
                }

                if (!cancelled) {
                    setIsSearching(false)
                    setTimeout(() => setSearchStatus(''), 4000)
                }
            } catch (error) {
                console.error('Error searching PDF:', error)
                if (!cancelled) {
                    setIsSearching(false)
                    setSearchStatus('')
                }
            }
        }

        searchAndHighlight()
        return () => { cancelled = true }
    }, [pdfDoc, citedChunk])

    const goToPrevPage = () => {
        if (currentPage > 1) {
            setCurrentPage(currentPage - 1)
            setHighlightBoxes([])
        }
    }

    const goToNextPage = () => {
        if (currentPage < totalPages) {
            setCurrentPage(currentPage + 1)
            setHighlightBoxes([])
        }
    }

    // Helper to strip chunk prefix for display
    function cleanChunkText(text) {
        let displayText = text.trim()
        const prefixMatch = displayText.match(/^.+ — .+\n\n/)
        if (prefixMatch) {
            displayText = displayText.slice(prefixMatch[0].length).trim()
        }
        const dLines = displayText.split('\n')
        let dStart = 0
        let dInCaption = false
        for (let i = 0; i < dLines.length; i++) {
            const dl = dLines[i].trim()
            if (!dl) { dInCaption = false; dStart = i + 1; continue }
            if (/^(Figure|Fig\.|Table|Algorithm)\s+\d+/i.test(dl)) { dInCaption = true; dStart = i + 1; continue }
            if (dInCaption && dl.length < 50 && /^[a-z]/.test(dl)) { dStart = i + 1; continue }
            dInCaption = false
            if (dl.length < 60 && !/[.!?:,]$/.test(dl)) { dStart = i + 1; continue }
            break
        }
        if (dStart > 0 && dStart < dLines.length) {
            displayText = dLines.slice(dStart).join('\n').trim()
        }
        return displayText
    }

    return (
        <div className={`paper-panel ${isEmpty ? 'empty' : 'visible'}`}>
            {/* Header bar */}
            <div className="paper-viewer-header">
                <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                        {!isEmpty && (
                            <button onClick={onClose}
                                className="p-1 rounded-lg flex-shrink-0 transition-colors"
                                style={{ color: 'var(--apple-text-secondary)' }}
                                title="Close">
                                <XIcon size={16} />
                            </button>
                        )}
                        <span className="text-xs truncate font-medium" style={{ color: 'var(--apple-text-primary)' }}>
                            {isEmpty ? 'Reference Viewer' : paper.title}
                        </span>
                    </div>
                    {!isEmpty && (
                        <div className="flex items-center gap-2 flex-shrink-0">
                            {paper.year > 0 && (
                                <span className="text-[11px]" style={{ color: 'var(--apple-text-quaternary)' }}>{paper.year}</span>
                            )}
                            {paper.arxiv_id && (
                                <a href={`https://arxiv.org/abs/${paper.arxiv_id}`}
                                    target="_blank" rel="noopener noreferrer"
                                    className="text-[11px]" style={{ color: 'var(--apple-accent)' }}>
                                    arXiv:{paper.arxiv_id}
                                </a>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Content area */}
            {isEmpty ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8" style={{ color: 'var(--apple-text-quaternary)' }}>
                    <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4" style={{ background: 'var(--apple-bg-tertiary)' }}>
                        <EyeIcon size={32} />
                    </div>
                    <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--apple-text-secondary)' }}>No reference selected</h3>
                    <p className="text-xs max-w-xs" style={{ color: 'var(--apple-text-quaternary)' }}>
                        Click the eye icon next to any source to view the full paper with highlighted citations.
                    </p>
                </div>
            ) : hasPdf && !pdfError && pdfDoc ? (
                <div className="paper-pdf-container" ref={containerRef}>
                    <div className="pdf-controls">
                        <button onClick={goToPrevPage} disabled={currentPage <= 1 || isSearching}>
                            &larr; Previous
                        </button>
                        <span className="pdf-status">
                            Page {currentPage} of {totalPages}
                            {isSearching && (
                                <span className="loading-dots">
                                    <span></span><span></span><span></span>
                                </span>
                            )}
                        </span>
                        <button onClick={goToNextPage} disabled={currentPage >= totalPages || isSearching}>
                            Next &rarr;
                        </button>
                        {searchStatus && (
                            <span className="text-xs ml-2" style={{ color: 'var(--apple-yellow)' }}>{searchStatus}</span>
                        )}
                    </div>

                    <div className="paper-pdf-content">
                        <div className="pdf-canvas-wrapper">
                            <canvas ref={canvasRef} className="pdf-canvas" />
                            {highlightBoxes.map((box, idx) => (
                                <div
                                    key={idx}
                                    className={`pdf-text-highlight ${box.isFirst ? 'first-highlight' : ''}`}
                                    style={{
                                        left: `${box.leftPct}%`,
                                        top: `${box.topPct}%`,
                                        width: `${box.widthPct}%`,
                                        height: `${box.heightPct}%`
                                    }}
                                />
                            ))}
                        </div>
                    </div>

                    {citedChunk && (
                        <div className="cited-passage-card">
                            <div className="cited-passage-header">
                                <div className="flex items-center gap-2">
                                    <span className="text-[10px] font-semibold" style={{ color: 'var(--apple-yellow)' }}>
                                        Referenced passage
                                    </span>
                                    {citedChunk.section && (
                                        <span className="cited-passage-section">
                                            {citedChunk.section.replace(/_/g, ' ')}
                                        </span>
                                    )}
                                </div>
                            </div>
                            <div className="cited-passage-body">
                                {cleanChunkText(citedChunk.text)}
                            </div>
                        </div>
                    )}
                </div>
            ) : hasPdf && !pdfError && !pdfDoc ? (
                <div className="flex-1 flex items-center justify-center text-sm" style={{ color: 'var(--apple-text-quaternary)' }}>
                    <div className="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                    <span className="ml-2">Loading PDF...</span>
                </div>
            ) : !isEmpty ? (
                <div className="flex-1 flex items-center justify-center text-sm p-8 text-center" style={{ color: 'var(--apple-text-quaternary)' }}>
                    <div>
                        <p className="mb-2">PDF not available for this paper.</p>
                        {paper && paper.arxiv_id && (
                            <a href={`https://arxiv.org/pdf/${paper.arxiv_id}`}
                                target="_blank" rel="noopener noreferrer"
                                className="text-xs" style={{ color: 'var(--apple-accent)' }}>
                                Open on arXiv directly
                            </a>
                        )}
                    </div>
                </div>
            ) : null}
        </div>
    )
}
