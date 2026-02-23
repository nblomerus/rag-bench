import React, { useState, useEffect } from 'react'
import { TabProvider, useTab } from './context/TabContext'
import { TabNavigation } from './components/TabNavigation'
import { AskTab } from './tabs/AskTab'
import { BenchmarksTab } from './tabs/BenchmarksTab'
import { ProductionTab } from './tabs/ProductionTab'
import { BookIcon, Spinner } from './components/Icons'
import { API_BASE } from './utils/api'

function AppContent() {
    const { activeTab } = useTab()
    const [ready, setReady] = useState(false)
    const [serverOffline, setServerOffline] = useState(false)

    useEffect(() => {
        let cancelled = false
        let retryTimeout = null

        async function checkHealth() {
            try {
                const res = await fetch(`${API_BASE}/health`)
                if (!cancelled) {
                    setReady(res.ok)
                    setServerOffline(!res.ok)
                    if (!res.ok) retryTimeout = setTimeout(checkHealth, 5000)
                }
            } catch {
                if (!cancelled) {
                    setReady(false)
                    setServerOffline(true)
                    retryTimeout = setTimeout(checkHealth, 5000)
                }
            }
        }

        checkHealth()
        return () => {
            cancelled = true
            if (retryTimeout) clearTimeout(retryTimeout)
        }
    }, [])

    return (
        <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
            {/* Header */}
            <header className="flex-shrink-0 flex items-center justify-between px-4 py-2 border-b border-gray-800">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-blue-600/10 border border-blue-600/20 rounded-lg flex items-center justify-center">
                        <BookIcon size={16} />
                    </div>
                    <h1 className="text-sm font-semibold text-gray-200">RAG-Bench</h1>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${serverOffline ? 'bg-red-500' : ready ? 'bg-green-500' : 'bg-yellow-500'}`}></span>
                        <span className="text-[10px] text-gray-500">
                            {serverOffline ? 'Offline' : ready ? 'Online' : 'Connecting...'}
                        </span>
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <TabNavigation />

            {/* Tab Content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Ask tab — always mounted, visibility toggled to preserve state */}
                <div className={`flex-1 flex ${activeTab === 'ask' ? '' : 'hidden'}`}>
                    <AskTab ready={ready} serverOffline={serverOffline} />
                </div>
                {activeTab === 'benchmarks' && <BenchmarksTab />}
                {activeTab === 'production' && <ProductionTab />}
            </div>
        </div>
    )
}

export default function App() {
    return (
        <TabProvider>
            <AppContent />
        </TabProvider>
    )
}
