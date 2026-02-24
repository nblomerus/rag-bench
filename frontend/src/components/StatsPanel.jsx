import React from 'react'
import { BookIcon, DatabaseIcon, CpuIcon, ZapIcon, SearchIcon, XIcon } from './Icons'

export function StatsPanel({ stats, onClose }) {
    if (!stats) return null
    const items = [
        { label: 'Papers', value: stats.total_papers, icon: BookIcon },
        { label: 'Chunks', value: stats.total_chunks?.toLocaleString(), icon: DatabaseIcon },
        { label: 'Embedding', value: (stats.embedding_model || '').split('/')[1], icon: CpuIcon },
        { label: 'LLM Backend', value: stats.llm_backend, icon: ZapIcon },
        { label: 'Model', value: (stats.llm_model || '').split(':')[0], icon: CpuIcon },
        { label: 'Collection', value: stats.collection_name, icon: SearchIcon },
    ]
    return (
        <div className="glass-card p-5 mb-4">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold flex items-center gap-2" style={{ color: 'var(--apple-text-primary)' }}>
                    <DatabaseIcon size={16} /> Pipeline Stats
                </h3>
                <button onClick={onClose} style={{ color: 'var(--apple-text-quaternary)' }} className="hover:opacity-80"><XIcon size={16} /></button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {items.map(({ label, value, icon: Ic }) => (
                    <div key={label} className="rounded-xl p-3" style={{ background: 'var(--apple-bg-tertiary)' }}>
                        <div className="flex items-center gap-1.5 text-[11px] mb-1" style={{ color: 'var(--apple-text-tertiary)' }}><Ic size={12} /> {label}</div>
                        <p className="text-sm font-medium" style={{ color: 'var(--apple-text-primary)' }}>{value}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}
