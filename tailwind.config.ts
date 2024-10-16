import type { Config } from 'tailwindcss';

export default {
  content: ["./src/**/*.{ts,tsx,civet}"],
  theme: {
    extend: {
      colors: {
        'bordeaux': '#9a2910'
      },

      height: {
        '30': '7.5rem',
      },
      backgroundColor: {
        board: '#2d4a32',
      },
      backgroundImage: {
        main: "url('../background.webp')",
        thinking: "url('../girl_thinking.webp')",
        happy: "url('../girl_happy.webp')",
        crying: "url('../girl_crying.webp')",
        speaking: "url('../girl_speaking.webp')",
        surprised: "url('../girl_surprised.webp')",
      },
      gridTemplateColumns: {
        '20/80': '20% 80%',
      },
      boxShadow: {
        picked: '0 0 20px rgba(0, 255, 255, 0.8), 0 0 30px rgba(0, 255, 255, 0.6), 0 0 40px rgba(0, 255, 255, 0.4)'
      },
      animation: {
        'flip-y': 'flip-y 500ms linear forwards',
        'lion-arrow': 'lion-arrow 2000ms linear forwards infinite',
      },
      keyframes: {
        "flip-y": {
          '0%': { opacity: '0', transform: 'rotateY(180deg)' },
          '100%': { opacity: '1', transform: 'rotateY(0)' },
        },
        "lion-arrow": {
          '0%, 100%': {opacity: '0.7'},
          '50%': {opacity: '0'    },
        }
      },

    }
  },
  plugins: [],
} satisfies Config