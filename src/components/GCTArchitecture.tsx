import React, { useState } from 'react';
import {
  Database,
  Server,
  Brain,
  Users,
  Shield,
  BarChart3,
  Zap,
  Cloud,
  GitBranch,
  Settings,
} from 'lucide-react';

type ComponentInfo = {
  title: string;
  icon: JSX.Element;
  tech: string;
  details: string[];
  color: string;
};

const GCTArchitecture: React.FC = () => {
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);

  const components: Record<string, ComponentInfo> = {
    frontend: {
      title: 'Frontend Layer',
      icon: <Users className="w-6 h-6" />,
      tech: 'React/Next.js, TypeScript, Tailwind CSS',
      details: [
        'Assessment Dashboard - Real-time coherence visualization',
        'Admin Portal - Organization management & analytics',
        'Mobile App - Continuous micro-assessments',
        'API Documentation - Developer portal',
      ],
      color: 'bg-blue-500',
    },
    gateway: {
      title: 'API Gateway',
      icon: <GitBranch className="w-6 h-6" />,
      tech: 'Kong/AWS API Gateway',
      details: [
        'Rate limiting & authentication',
        'Request routing & load balancing',
        'API versioning & deprecation',
        'Monitoring & analytics',
      ],
      color: 'bg-purple-500',
    },
    auth: {
      title: 'Authentication & Authorization',
      icon: <Shield className="w-6 h-6" />,
      tech: 'Auth0/Cognito + RBAC',
      details: [
        'SSO integration (SAML, OIDC)',
        'Multi-tenant organization isolation',
        'Role-based access control',
        'GDPR/HIPAA compliance',
      ],
      color: 'bg-red-500',
    },
    core: {
      title: 'Core Services',
      icon: <Server className="w-6 h-6" />,
      tech: 'Python/FastAPI, Docker, Kubernetes',
      details: [
        'Assessment Service - Scenario generation & scoring',
        'Coherence Engine - Mathematical computations',
        'Network Analysis - Transmission dynamics',
        'Notification Service - Real-time updates',
      ],
      color: 'bg-green-500',
    },
    ai: {
      title: 'AI/ML Layer',
      icon: <Brain className="w-6 h-6" />,
      tech: 'Grok 3 (GitHub Models), PyTorch',
      details: [
        'Scenario Generation - Cultural adaptation',
        'Pattern Recognition - Coherence transitions',
        'Predictive Analytics - Leadership emergence',
        'NLP Processing - Qualitative assessment',
      ],
      color: 'bg-orange-500',
    },
    compute: {
      title: 'Computation Engine',
      icon: <Zap className="w-6 h-6" />,
      tech: 'Celery, Redis, NumPy/SciPy',
      details: [
        'Biological Optimization - q^optimal calculations',
        'Derivative Analysis - dC/dt tracking (time-aware)',
        'Async Streaming - aiokafka consumers',
        'Network Algorithms - Transmission modeling',
        'Statistical Analysis - Validation & reporting',
      ],
      color: 'bg-yellow-500',
    },
    database: {
      title: 'Data Layer',
      icon: <Database className="w-6 h-6" />,
      tech: 'PostgreSQL, TimescaleDB, Redis',
      details: [
        'User Data - Profiles & organizations',
        'Time Series - Coherence tracking',
        'Network Data - Relationship mapping',
        'Caching - Performance optimization',
      ],
      color: 'bg-indigo-500',
    },
    analytics: {
      title: 'Analytics & Monitoring',
      icon: <BarChart3 className="w-6 h-6" />,
      tech: 'DataDog, Grafana, ELK Stack',
      details: [
        'Real-time Metrics - System performance',
        'Business Analytics - Usage patterns',
        'Coherence Insights - Trend analysis',
        'Alerting - Anomaly detection',
      ],
      color: 'bg-pink-500',
    },
    infrastructure: {
      title: 'Infrastructure',
      icon: <Cloud className="w-6 h-6" />,
      tech: 'AWS/Azure, Terraform, CI/CD',
      details: [
        'Auto-scaling - Demand-based compute',
        'Multi-region - Global deployment',
        'Backup & Recovery - Data protection',
        'Security - Network isolation & encryption',
      ],
      color: 'bg-gray-500',
    },
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gray-50 rounded-lg">
      <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
        GCT SaaS Technical Architecture
      </h1>
      {/* Architecture Diagram */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {/* Top Row - User Interface */}
        <div className="col-span-3 flex justify-center">
          <ComponentBox
            component={components.frontend}
            selected={selectedComponent === 'frontend'}
            onClick={() =>
              setSelectedComponent(
                selectedComponent === 'frontend' ? null : 'frontend'
              )
            }
          />
        </div>
        {/* Second Row - Gateway & Auth */}
        <div className="flex justify-center">
          <ComponentBox
            component={components.gateway}
            selected={selectedComponent === 'gateway'}
            onClick={() =>
              setSelectedComponent(
                selectedComponent === 'gateway' ? null : 'gateway'
              )
            }
          />
        </div>
        <div className="flex justify-center">
          <ComponentBox
            component={components.auth}
            selected={selectedComponent === 'auth'}
            onClick={() =>
              setSelectedComponent(selectedComponent === 'auth' ? null : 'auth')
            }
          />
        </div>
        <div className="flex justify-center">
          <ComponentBox
            component={components.analytics}
            selected={selectedComponent === 'analytics'}
            onClick={() =>
              setSelectedComponent(
                selectedComponent === 'analytics' ? null : 'analytics'
              )
            }
          />
        </div>
        {/* Third Row - Core Services */}
        <div className="flex justify-center">
          <ComponentBox
            component={components.core}
            selected={selectedComponent === 'core'}
            onClick={() =>
              setSelectedComponent(selectedComponent === 'core' ? null : 'core')
            }
          />
        </div>
        <div className="flex justify-center">
          <ComponentBox
            component={components.ai}
            selected={selectedComponent === 'ai'}
            onClick={() =>
              setSelectedComponent(selectedComponent === 'ai' ? null : 'ai')
            }
          />
        </div>
        <div className="flex justify-center">
          <ComponentBox
            component={components.compute}
            selected={selectedComponent === 'compute'}
            onClick={() =>
              setSelectedComponent(
                selectedComponent === 'compute' ? null : 'compute'
              )
            }
          />
        </div>
        {/* Bottom Row - Infrastructure */}
        <div className="flex justify-center">
          <ComponentBox
            component={components.database}
            selected={selectedComponent === 'database'}
            onClick={() =>
              setSelectedComponent(
                selectedComponent === 'database' ? null : 'database'
              )
            }
          />
        </div>
        <div className="col-span-2 flex justify-center">
          <ComponentBox
            component={components.infrastructure}
            selected={selectedComponent === 'infrastructure'}
            onClick={() =>
              setSelectedComponent(
                selectedComponent === 'infrastructure' ? null : 'infrastructure'
              )
            }
          />
        </div>
      </div>
      {/* Component Details */}
      {selectedComponent && (
        <div className="mt-8 p-6 bg-white rounded-lg shadow-lg border-l-4 border-blue-500">
          <div className="flex items-center gap-3 mb-4">
            <div
              className={`p-2 rounded-lg ${components[selectedComponent].color} text-white`}
            >
              {components[selectedComponent].icon}
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-800">
                {components[selectedComponent].title}
              </h3>
              <p className="text-sm text-gray-600">
                {components[selectedComponent].tech}
              </p>
            </div>
          </div>
          <ul className="space-y-2">
            {components[selectedComponent].details.map((detail, index) => (
              <li key={index} className="flex items-start gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                <span className="text-gray-700">{detail}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {/* Key Technical Considerations */}
      <div className="mt-8 grid md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">
            Scalability Patterns
          </h3>
          <ul className="space-y-2 text-sm text-gray-600">
            <li>• Microservices with async message queues</li>
            <li>• Horizontal scaling for computation-heavy tasks</li>
            <li>• Database sharding by organization</li>
            <li>• CDN for global content delivery</li>
          </ul>
        </div>
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">
            Security & Compliance
          </h3>
          <ul className="space-y-2 text-sm text-gray-600">
            <li>• End-to-end encryption for sensitive data</li>
            <li>• SOC 2 Type II compliance</li>
            <li>• GDPR data portability & deletion</li>
            <li>• Regular security audits & penetration testing</li>
          </ul>
        </div>
      </div>
      <div className="mt-6 text-center text-sm text-gray-500">
        Click on components above to see implementation details
      </div>
    </div>
  );
};

interface ComponentBoxProps {
  component: ComponentInfo;
  selected: boolean;
  onClick: () => void;
}

const ComponentBox: React.FC<ComponentBoxProps> = ({
  component,
  selected,
  onClick,
}) => (
  <div
    className={`p-4 rounded-lg cursor-pointer transition-all duration-200 border-2 min-w-40 text-center
      ${selected ? 'border-blue-500 bg-blue-50 shadow-lg scale-105' :
        'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md'}`}
    onClick={onClick}
  >
    <div className={`inline-flex p-2 rounded-lg mb-2 ${component.color} text-white`}>
      {component.icon}
    </div>
    <h4 className="font-semibold text-gray-800 text-sm">
      {component.title}
    </h4>
    <p className="text-xs text-gray-600 mt-1">
      {component.tech}
    </p>
  </div>
);

export default GCTArchitecture;
