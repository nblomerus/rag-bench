/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Display"', '"SF Pro Text"', 'system-ui', 'sans-serif'],
      },
      colors: {
        gray: {
          950: '#0a0f1a',
          900: '#111827',
          850: '#151d2e',
          800: '#1e293b',
          700: '#334155',
          600: '#475569',
          500: '#64748b',
          400: '#94a3b8',
          300: '#cbd5e1',
        },
        apple: {
          bg: {
            primary: 'var(--apple-bg-primary)',
            secondary: 'var(--apple-bg-secondary)',
            tertiary: 'var(--apple-bg-tertiary)',
            elevated: 'var(--apple-bg-elevated)',
          },
          text: {
            primary: 'var(--apple-text-primary)',
            secondary: 'var(--apple-text-secondary)',
            tertiary: 'var(--apple-text-tertiary)',
            quaternary: 'var(--apple-text-quaternary)',
          },
          accent: 'var(--apple-accent)',
          green: 'var(--apple-green)',
          red: 'var(--apple-red)',
          yellow: 'var(--apple-yellow)',
          orange: 'var(--apple-orange)',
        },
      },
      boxShadow: {
        'apple-sm': 'var(--apple-shadow-sm)',
        'apple-md': 'var(--apple-shadow-md)',
        'apple-lg': 'var(--apple-shadow-lg)',
      },
      borderRadius: {
        'apple': '12px',
        'apple-lg': '16px',
        'apple-xl': '20px',
      },
    },
  },
  plugins: [],
}
