import { Box, Grid, Typography } from '@mui/material';
import BlockRenderer from './BlockRenderer';
import { shouldShowPageSubtitle } from './displayUtils';
import { PageFilterProvider } from './PageFilterContext';
import { useAppDefinition } from './useAppDefinition';

type Block = Record<string, unknown>;

const CHART_TYPES = new Set(['lineChart', 'barChart', 'pieChart']);

function isChartBlock(block: Block): boolean {
  return CHART_TYPES.has(String(block.type));
}

type Segment =
  | { kind: 'single'; block: Block; index: number }
  | { kind: 'chartGrid'; blocks: Block[]; startIndex: number };

function segmentBlocks(blocks: Block[]): Segment[] {
  const segments: Segment[] = [];
  let i = 0;
  while (i < blocks.length) {
    const block = blocks[i];
    if (!isChartBlock(block)) {
      segments.push({ kind: 'single', block, index: i });
      i += 1;
      continue;
    }
    const run: Block[] = [];
    const start = i;
    while (i < blocks.length && isChartBlock(blocks[i])) {
      run.push(blocks[i]);
      i += 1;
    }
    segments.push({ kind: 'chartGrid', blocks: run, startIndex: start });
  }
  return segments;
}

export default function PageRenderer({ pageId }: { pageId: string }) {
  const { definition } = useAppDefinition();
  const page = definition.pages.find((p) => p.id === pageId);

  if (!page) {
    return <Typography color="error">Unknown page: {pageId}</Typography>;
  }

  const segments = segmentBlocks(page.blocks);
  const showSubtitle = shouldShowPageSubtitle(page.subtitle, page.title);

  return (
    <PageFilterProvider>
      <Box sx={{ maxWidth: 1440, mx: 'auto' }}>
        {showSubtitle ? (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2.5 }}>
            {page.subtitle}
          </Typography>
        ) : null}
        {segments.map((seg) => {
          if (seg.kind === 'single') {
            return (
              <BlockRenderer
                key={`${pageId}-${seg.index}`}
                block={seg.block}
                definition={definition}
                pageId={pageId}
                pageTitle={page.title}
                pageBlocks={page.blocks}
              />
            );
          }
          const multi = seg.blocks.length > 1;
          return (
            <Grid container spacing={2.5} sx={{ mb: 3 }} key={`${pageId}-charts-${seg.startIndex}`}>
              {seg.blocks.map((block, j) => (
                <Grid
                  item
                  xs={12}
                  md={multi ? 6 : 12}
                  lg={multi && seg.blocks.length >= 3 ? 4 : multi ? 6 : 12}
                  key={`${pageId}-${seg.startIndex + j}`}
                >
                  <BlockRenderer
                    block={block}
                    definition={definition}
                    pageId={pageId}
                    pageTitle={page.title}
                    pageBlocks={page.blocks}
                    layout="grid"
                  />
                </Grid>
              ))}
            </Grid>
          );
        })}
      </Box>
    </PageFilterProvider>
  );
}
