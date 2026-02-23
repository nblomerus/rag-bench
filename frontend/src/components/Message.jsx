import React, { useState, useRef, useEffect } from 'react'
import { formatAnswer } from '../utils/render'
import { SourceCard } from './SourceCard'
import { QualityBadges } from './metrics/QualityBadges'
import { QualityPanel } from './metrics/QualityPanel'
import { Tip } from './metrics/Tip'
import { AlertIcon, ZapIcon, CpuIcon, SearchIcon, ChevronDown, ChevronUp } from './Icons'

export function Message({ message, onViewSource }) {
    const [expandedSources, setExpandedSources] = useState({})
    const [showQuality, setShowQuality] = useState(false)
    const toggleSource = (idx) => setExpandedSources(p => ({ ...p, [idx]: !p[idx] }))
    const answerRef = useRef(null)

    // Handle clicks on [Source N] citation refs inside the rendered answer
    useEffect(() => {
        const el = answerRef.current
        if (!el || !onViewSource || !message.data?.sources?.length) return
        const handler = (e) => {
            const citEl = e.target.closest('[data-source-idx]')
            if (!citEl) return
            const idx = parseInt(citEl.dataset.sourceIdx, 10) - 1
            const source = message.data.sources[idx]
            if (source) onViewSource(source)
        }
        el.addEventListener('click', handler)
        return () => el.removeEventListener('click', handler)
    }, [message.data?.sources, onViewSource])

    if (message.role === 'user') {
        return (
            <div className="flex justify-end mb-4">
                <div className="bg-blue-600 text-white px-4 py-3 rounded-2xl rounded-br-md max-w-[75%]">
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                </div>
            </div>
        )
    }

    const { data, streaming } = message
    const answerText = (message.content || '').split('\nSources:\n')[0]
    const quality = data?.quality
    const hasQuality = quality && quality.retrieval_confidence !== 'unknown' && !data?.deflected

    return (
        <div className="flex justify-start mb-6">
            <div className="max-w-[85%] w-full">
                {data?.deflected && !streaming && (
                    <div className="flex items-center gap-2 bg-amber-900/30 border border-amber-700/50 text-amber-300 px-3 py-2 rounded-lg mb-2 text-xs">
                        <AlertIcon size={14} />
                        <span>Deflected: {data.deflection_reason}</span>
                    </div>
                )}
                <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-md px-4 py-3">
                    <div ref={answerRef} className="text-sm text-gray-200 leading-relaxed">
                        {answerText ? formatAnswer(answerText) : (streaming ? '' : 'No response.')}
                        {streaming && <span className="inline-block w-2 h-4 bg-blue-400 ml-0.5 animate-pulse rounded-sm" />}
                    </div>
                    {data && !streaming && (
                        <div className="mt-3 pt-2 border-t border-gray-700 space-y-2">
                            <div className="flex items-center gap-3 text-xs text-gray-500">
                                <Tip id="latency"><span className="flex items-center gap-1"><ZapIcon size={12} />{data.latency_ms?.toFixed(0) || '0'}ms</span></Tip>
                                <Tip id="backend_model"><span className="flex items-center gap-1"><CpuIcon size={12} />{data.backend || '...'}/{((data.model || '').split(':')[0]) || '...'}</span></Tip>
                                <Tip id="sources_count"><span className="flex items-center gap-1"><SearchIcon size={12} />{data.sources?.length || 0} relevant source{data.sources?.length !== 1 ? 's' : ''}</span></Tip>
                            </div>
                            {hasQuality && (
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <QualityBadges quality={quality} />
                                    <button
                                        onClick={() => setShowQuality(v => !v)}
                                        style={{ marginLeft: 'auto', fontSize: '11px', color: '#6b7280', display: 'flex', alignItems: 'center', gap: '2px', background: 'none', border: 'none', cursor: 'pointer', padding: '0' }}
                                        onMouseOver={e => e.currentTarget.style.color = '#d1d5db'}
                                        onMouseOut={e => e.currentTarget.style.color = '#6b7280'}
                                    >
                                        {showQuality ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                                        <span>Details</span>
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                    {streaming && data?.sources?.length > 0 && (
                        <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-700 text-xs text-gray-500">
                            <SearchIcon size={12} />
                            <span>{data.sources.length} source{data.sources.length !== 1 ? 's' : ''} found — generating answer...</span>
                        </div>
                    )}
                </div>
                {showQuality && hasQuality && <QualityPanel quality={quality} />}
                {data?.sources?.length > 0 && !data.deflected && (
                    <div className="mt-2 space-y-1.5">
                        {data.sources.map((source, idx) => (
                            <SourceCard key={idx} source={source} index={idx}
                                isExpanded={expandedSources[idx] || false}
                                onToggle={() => toggleSource(idx)}
                                onViewInPaper={onViewSource} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
