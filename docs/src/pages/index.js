import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <img
          className={styles.heroLogo}
          src={useBaseUrl('/img/logo.png')}
          alt="mograd logo"
          width="140"
          height="140"
        />
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <p className={styles.status}>
          Early-stage and not usable as a dependency. No released package, no API
          stability.
        </p>
        <div className={styles.buttons}>
          <Link
            className={clsx('button button--outline button--lg', styles.buttonNeutral)}
            to="/docs/api/">
            API Reference
          </Link>
          <Link
            className="button button--outline button--primary button--lg"
            to="/docs/contribute/contributing">
            Contribute
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <Hero />
    </Layout>
  );
}
