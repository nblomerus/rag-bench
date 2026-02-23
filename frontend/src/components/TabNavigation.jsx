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
        <div className="flex border-b border-gray-800 bg-gray-900/60 px-4">
            {tabs.map(tab => (
                <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium transition-colors relative ${
                        activeTab === tab.id
                            ? 'text-blue-400'
                            : 'text-gray-500 hover:text-gray-300'
                    }`}
                >
                    <tab.icon size={14} />
                    {tab.label}
                    {activeTab === tab.id && (
                        <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500 rounded-t" />
                    )}
                </button>
            ))}
        </div>
    )
}
