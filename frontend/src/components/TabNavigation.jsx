import React from 'react'
import { BookIcon, BarChartIcon, CpuIcon } from './Icons'
import { useTab } from '../context/TabContext'

const tabs = [
    { id: 'ask', label: 'Ask', icon: BookIcon },
    { id: 'benchmarks', label: 'Benchmarks', icon: BarChartIcon },
    { id: 'production', label: 'Production', icon: CpuIcon },
]

export function TabNavigation() {
    const { activeTab, setActiveTab } = useTab()

    return (
        <div className="flex border-b apple-divider px-6" style={{ background: 'var(--apple-glass-bg)', backdropFilter: 'blur(20px) saturate(180%)' }}>
            {tabs.map(tab => (
                <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className="flex items-center gap-2 px-4 py-2.5 text-xs font-medium transition-colors relative"
                    style={{
                        color: activeTab === tab.id ? 'var(--apple-accent)' : 'var(--apple-text-quaternary)',
                    }}
                    onMouseEnter={e => { if (activeTab !== tab.id) e.currentTarget.style.color = 'var(--apple-text-secondary)' }}
                    onMouseLeave={e => { if (activeTab !== tab.id) e.currentTarget.style.color = 'var(--apple-text-quaternary)' }}
                >
                    <tab.icon size={14} />
                    {tab.label}
                    {activeTab === tab.id && (
                        <span className="absolute bottom-0 left-0 right-0 h-[2px] rounded-full" style={{ background: 'var(--apple-accent)' }} />
                    )}
                </button>
            ))}
        </div>
    )
}
