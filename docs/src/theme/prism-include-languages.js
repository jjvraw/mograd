/**
 * Swizzled from @docusaurus/theme-classic to register a `mojo` Prism language.
 *
 * Prism has no Mojo grammar. Mojo's surface syntax is close enough to Python
 * that reusing that grammar highlights correctly, with the Mojo-only keywords
 * layered on top.
 */
import siteConfig from '@generated/docusaurus.config';

const MOJO_KEYWORDS =
  /\b(?:alias|comptime|fn|struct|trait|var|owned|borrowed|inout|mut|read|out|ref|deinit|raises|capturing|parameter|__type_of|__origin_of)\b/;

export default function prismIncludeLanguages(PrismObject) {
  const {
    themeConfig: {prism},
  } = siteConfig;
  const {additionalLanguages} = prism;

  const PrismBefore = globalThis.Prism;
  globalThis.Prism = PrismObject;

  additionalLanguages.forEach((lang) => {
    if (lang === 'php') {
      require('prismjs/components/prism-markup-templating.js');
    }
    require(`prismjs/components/prism-${lang}`);
  });

  if (PrismObject.languages.python) {
    PrismObject.languages.mojo = PrismObject.languages.extend('python', {});
    PrismObject.languages.insertBefore('mojo', 'keyword', {
      'mojo-keyword': {
        pattern: MOJO_KEYWORDS,
        alias: 'keyword',
      },
    });
  }

  delete globalThis.Prism;
  if (typeof PrismBefore !== 'undefined') {
    globalThis.Prism = PrismObject;
  }
}
