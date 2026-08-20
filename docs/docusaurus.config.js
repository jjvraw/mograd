// @ts-check

/**
 * Syntax themes built from the site palette.  
*/
const darkCodeTheme = {
  plain: {color: '#CEC19D', backgroundColor: '#292828'},
  styles: [
    {types: ['comment'], style: {color: '#9A917B', fontStyle: 'italic'}},
    {types: ['punctuation', 'operator'], style: {color: '#A4997B'}},
    {types: ['keyword', 'mojo-keyword', 'boolean', 'decorator'], style: {color: '#FF7C0A'}},
    {types: ['string', 'char', 'number', 'constant'], style: {color: '#FF4C1F'}},
    {types: ['function', 'class-name', 'builtin'], style: {color: '#FFFFFF'}},
  ],
};

const lightCodeTheme = {
  plain: {color: '#1A1A1A', backgroundColor: '#FAF8F3'},
  styles: [
    {types: ['comment'], style: {color: '#6E6E6E', fontStyle: 'italic'}},
    {types: ['punctuation', 'operator'], style: {color: '#5A5A5A'}},
    {types: ['keyword', 'mojo-keyword', 'boolean', 'decorator'], style: {color: '#9E4A00'}},
    {types: ['string', 'char', 'number', 'constant'], style: {color: '#C22700'}},
    {types: ['function', 'class-name', 'builtin'], style: {color: '#000000'}},
  ],
};

const organizationName = 'jjvraw';
const projectName = 'mograd';
const repoUrl = `https://github.com/${organizationName}/${projectName}`;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'mograd',
  tagline: 'An autograd tensor library for Mojo',
  favicon: 'img/logo.png',

  future: {
    v4: true,
  },

  url: `https://${organizationName}.github.io`,
  baseUrl: `/${projectName}/`,

  organizationName,
  projectName,

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    // Generated API pages are plain CommonMark `.md`, so they need no MDX
    // escaping. Hand-written `.mdx` pages keep full JSX support.
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
    remarkRehypeOptions: {
      // Replaces the default `↩` on footnote backreferences. The index is only
      // appended when a footnote is cited more than once.
      footnoteBackContent: (_, rereferenceIndex) => {
        const content = [{type: 'text', value: '^'}];
        if (rereferenceIndex > 1) {
          content.push({
            type: 'element',
            tagName: 'sup',
            properties: {},
            children: [{type: 'text', value: String(rereferenceIndex)}],
          });
        }
        return content;
      },
    },
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: `${repoUrl}/tree/main/docs/`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  // Offline search. Algolia DocSearch needs an approved application and a
  // crawlable public URL, so the index is built locally at build time instead.
  themes: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        indexBlog: false,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        searchResultLimits: 8,
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/logo.png',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      // Generated API pages nest methods under a type's "Methods" heading, so
      // the contents panel has to reach h4 to list them.
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },
      navbar: {
        title: 'mograd',
        logo: {
          alt: 'mograd logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'apiSidebar',
            position: 'left',
            label: 'API',
          },
          {
            type: 'docSidebar',
            sidebarId: 'contributeSidebar',
            position: 'left',
            label: 'Contribute',
          },
          {type: 'search', position: 'right'},
          {
            href: repoUrl,
            position: 'right',
            className: 'navbar-github-link',
            'aria-label': 'GitHub repository',
          },
        ],
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['python', 'bash', 'toml'],
      },
    }),
};

export default config;
