export default defineAppConfig({
  ui: {
    colors: { primary: 'indigo', neutral: 'slate' },
    // Keep solid actions readable while hovered or pressed in either theme.
    button: {
      compoundVariants: [
        { color: 'primary', variant: 'solid', class: 'hover:bg-primary/90 active:bg-primary/90' },
        { color: 'error', variant: 'solid', class: 'hover:bg-error/90 active:bg-error/90' },
      ],
    },
  },
})
