import React from 'react'

export function LoadingDots() {
    return (
        <div className="flex justify-start mb-4">
            <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-md px-4 py-3">
                <div className="loading-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
    )
}
