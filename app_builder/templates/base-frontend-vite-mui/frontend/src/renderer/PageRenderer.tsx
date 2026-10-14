import { Box, Typography } from '@mui/material';
import BlockRenderer from './BlockRenderer';
import { PageFilterProvider } from './PageFilterContext';
import { useAppDefinition } from './useAppDefinition';

export default function PageRenderer({ pageId }: { pageId: string }) {
  const { definition } = useAppDefinition();
  const page = definition.pages.find((p) => p.id === pageId);

  if (!page) {
    return <Typography color="error">Unknown page: {pageId}</Typography>;
  }

  return (
    <PageFilterProvider>
      <Box>
        <Typography variant="h4" fontWeight={700} gutterBottom>
          {page.title}
        </Typography>
        {page.subtitle ? (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            {page.subtitle}
          </Typography>
        ) : null}
        {page.blocks.map((block, i) => (
          <BlockRenderer
            key={`${pageId}-${i}`}
            block={block}
            definition={definition}
            pageId={pageId}
            pageBlocks={page.blocks}
          />
        ))}
      </Box>
    </PageFilterProvider>
  );
}
