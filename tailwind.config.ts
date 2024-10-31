import type { Config } from 'tailwindcss';

export default {
  content: ["./src/**/*.{ts,tsx,civet}"],
  theme: {
    extend: {
      height: {
        '30': '7.5rem',
      },
      backgroundColor: {
        board: '#2d4a32',
      },
      backgroundImage: {
        main: "linear-gradient(rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.3)), url('../background.webp')",
        thinking: "url('../girl_thinking.webp')",
        happy: "url('../girl_happy.webp')",
        crying: "url('../girl_crying.webp')",
        speaking: "url('../girl_speaking.webp')",
        surprised: "url('../girl_surprised.webp')",
        backface: 'radial-gradient(100.40% 90% at 95% 5%, #9a2910 0%, #600000 100%)',
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
        'oya-card': 'oya-card 500ms linear forwards infinite'
      },
      keyframes: {
        "flip-y": {
          '0%': { opacity: '0', transform: 'rotateY(180deg)' },
          '100%': { opacity: '1', transform: 'rotateY(0)' },
        },
        "lion-arrow": {
          '0%, 100%': {opacity: '0.7'},
          '50%': {opacity: '0'    },
        },
        "oya-card": {
          '0%, 100%': {transform: 'rotateY(180deg) scale(1.1)'},
          '50%': {transform: 'rotateY(180deg) scale(1)'},
        }
      },

    }
  },
  plugins: [],
} satisfies Config