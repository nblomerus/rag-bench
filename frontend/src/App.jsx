import React, { useState, useEffect } from 'react'
import { TabProvider, useTab } from './context/TabContext'
import { TabNavigation } from './components/TabNavigation'
import { AskTab } from './tabs/AskTab'
import { BenchmarksTab } from './tabs/BenchmarksTab'
import { ProductionTab } from './tabs/ProductionTab'
import { RLogoIcon, GitHubIcon, Spinner } from './components/Icons'
import { ThemeToggle } from './components/ThemeToggle'
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
        <div className="h-screen flex flex-col" style={{ background: 'var(--apple-bg-primary)', color: 'var(--apple-text-primary)' }}>
            {/* Header */}
            <header className="flex-shrink-0 flex items-center justify-between px-6 py-3 glass-thick border-b apple-divider sticky top-0 z-30">
                <div className="flex items-center gap-3">
                    <RLogoIcon size={32} />
                    <div>
                        <h1 className="text-[15px] font-semibold tracking-tight" style={{ color: 'var(--apple-text-primary)' }}>RAG-Bench</h1>
                        <p className="text-[11px]" style={{ color: 'var(--apple-text-tertiary)' }}>RAG Pipeline Benchmark</p>
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full`} style={{ background: serverOffline ? 'var(--apple-red)' : ready ? 'var(--apple-green)' : 'var(--apple-yellow)' }}></span>
                        <span className="text-[11px]" style={{ color: 'var(--apple-text-quaternary)' }}>
                            {serverOffline ? 'Offline' : ready ? 'Online' : 'Connecting...'}
                        </span>
                    </div>
                    <a
                        href="https://github.com/nblomerus/rag-bench"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center justify-center w-8 h-8 rounded-lg transition-colors"
                        style={{ color: 'var(--apple-text-tertiary)' }}
                        title="View on GitHub"
                    >
                        <GitHubIcon size={18} />
                    </a>
                    <ThemeToggle />
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
