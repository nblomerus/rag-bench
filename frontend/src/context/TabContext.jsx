import React, { createContext, useState, useEffect, useContext } from 'react'

const TabContext = createContext()

const VALID_TABS = ['ask', 'benchmarks', 'production']

export function TabProvider({ children }) {
    const [activeTab, setActiveTab] = useState(() => {
        const hash = window.location.hash.slice(1) || 'ask'
        return VALID_TABS.includes(hash) ? hash : 'ask'
    })

    // Sync state to URL hash for bookmarkability
    useEffect(() => {
        window.location.hash = activeTab
    }, [activeTab])

    // Listen for browser back/forward
    useEffect(() => {
        const handler = () => {
            const hash = window.location.hash.slice(1) || 'ask'
            if (VALID_TABS.includes(hash)) setActiveTab(hash)
        }
        window.addEventListener('hashchange', handler)
        return () => window.removeEventListener('hashchange', handler)
    }, [])

    return (
        <TabContext.Provider value={{ activeTab, setActiveTab }}>
            {children}
        </TabContext.Provider>
    )
}

export function useTab() {
    const ctx = useContext(TabContext)
    if (!ctx) throw new Error('useTab must be used within TabProvider')
    return ctx
}
