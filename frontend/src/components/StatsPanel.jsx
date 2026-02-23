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
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 mb-4">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                    <DatabaseIcon size={16} /> Pipeline Stats
                </h3>
                <button onClick={onClose} className="text-gray-500 hover:text-gray-300"><XIcon size={16} /></button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {items.map(({ label, value, icon: Ic }) => (
                    <div key={label} className="bg-gray-900 rounded-lg p-2">
                        <div className="flex items-center gap-1.5 text-gray-500 text-xs mb-1"><Ic size={12} /> {label}</div>
                        <p className="text-sm font-medium text-gray-200">{value}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}
