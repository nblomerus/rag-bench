import React from 'react'

// Use CDN globals (loaded via index.html script tags)
const katex = window.katex
const marked = window.marked

// Render math with KaTeX (safe fallback)
export function renderKatex(latex, displayMode) {
    try {
        return katex.renderToString(latex, { displayMode, throwOnError: false, trust: true })
    } catch (e) {
        return `<code class="text-yellow-300">${latex}</code>`
    }
}

// Format answer: Markdown + KaTeX + Citation refs
export function formatAnswer(text) {
    if (!text) return null

    // Step 1: Protect [Source N] citations from markdown mangling
    let processed = text.replace(/\[Source (\d+)\]/g, '%%SOURCE_$1%%')

    // Step 2: Protect and render KaTeX math BEFORE markdown processing
    const mathBlocks = []
    let mathIdx = 0

    // Display math: $$ ... $$
    processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_, latex) => {
        const rendered = renderKatex(latex.trim(), true)
        const placeholder = `%%MATH_BLOCK_${mathIdx}%%`
        mathBlocks.push(rendered)
        mathIdx++
        return placeholder
    })

    // Inline math: $ ... $ (but not $$)
    processed = processed.replace(/(?<!\$)\$(?!\$)((?:[^$\\]|\\.)+?)\$(?!\$)/g, (_, latex) => {
        const rendered = renderKatex(latex.trim(), false)
        const placeholder = `%%MATH_BLOCK_${mathIdx}%%`
        mathBlocks.push(rendered)
        mathIdx++
        return placeholder
    })

    // Step 3: Render markdown via marked
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false,
    })
    processed = marked.parse(processed)

    // Step 4: Restore math blocks (they were protected from markdown)
    for (let i = 0; i < mathBlocks.length; i++) {
        processed = processed.replace(`%%MATH_BLOCK_${i}%%`, mathBlocks[i])
    }

    // Step 5: Restore citation refs with clickable spans
    processed = processed.replace(/%%SOURCE_(\d+)%%/g,
        '<span class="citation-ref" data-source-idx="$1" role="button" tabindex="0">[Source $1]</span>')

    return <div className="answer-md" dangerouslySetInnerHTML={{ __html: processed }} />
}
