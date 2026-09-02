import Badge from '../ui/Badge';

const RISK_MAP = {
  Low:      { variant: 'success', dot: true },
  Moderate: { variant: 'warning', dot: true },
  Critical: { variant: 'danger',  dot: true },
};

export default function RiskBadge({ level }) {
  if (!level) return <Badge variant="default">—</Badge>;
  const { variant, dot } = RISK_MAP[level] ?? { variant: 'default', dot: false };
  return <Badge variant={variant} dot={dot}>{level}</Badge>;
}
