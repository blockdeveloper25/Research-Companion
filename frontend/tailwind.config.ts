import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Design tokens — light palette
        surface:  '#ffffff',   // page background
        sidebar:  '#f7f7f7',   // sidebar background
        raised:   '#f0f0f0',   // cards, input, assistant bubble
        border:   '#e5e5e5',   // all borders
        muted:    '#888888',   // secondary text
        dim:      '#bbbbbb',   // tertiary text / placeholders
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
} satisfies Config
