import React from 'react';
import { Container } from '@radix-ui/themes';
import { Flex } from '@radix-ui/themes';
import { Button } from '@radix-ui/themes';
import { Text, Heading } from '@radix-ui/themes';
import { Card } from '@radix-ui/themes';
import { Separator } from '@radix-ui/themes';
import { 
  ArrowRight, 
  Server, 
  Shield, 
  Lock, 
  Cpu, 
  BarChart, 
  Network 
} from 'lucide-react';
import HeroIllustration from './HeroIllustration';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-[#1a1a2e] text-white">
      {/* Navigation */}
      <nav className="absolute pl-[1%] pr-[1%] w-full bg-[#1a1a2e]/80 backdrop-blur-sm border-b border-white/10 z-50">
        <Container size="4">
          <div className="h-16 flex items-center justify-between">
            <Flex align="center" gap="4">
              <img src="/public/favlogo.png" alt="Compunir Logo" className="w-8 h-8" />
              <Text weight="bold" size="4" className="text-white">Compunir</Text>
            </Flex>
            <Flex gap="6" className="items-center">
              <a href="#features" className="text-white/70 hover:text-white transition">Features</a>
              <a href="#how-it-works" className="text-white/70 hover:text-white transition">How It Works</a>
              <a href="#network" className="text-white/70 hover:text-white transition">Network</a>
              <a href="https://github.com/ameritusweb/compunir" 
                 target="_blank" 
                 rel="noopener noreferrer" 
                 className="text-white/70 hover:text-white transition">
                GitHub
              </a>
              <Button className="bg-blue-500 hover:bg-blue-600 transition text-white px-4 py-2 rounded-md">
                Get Started
              </Button>
            </Flex>
          </div>
        </Container>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen pt-16 overflow-hidden">
      <div className="absolute inset 0 w-full h-full">
          <div className="absolute top-20 left-20 w-72 h-72 bg-blue-500 rounded-full opacity-10 blur-3xl animate-pulse"></div>
          <div className="absolute bottom-20 right-20 w-96 h-96 bg-purple-500 rounded-full opacity-10 blur-3xl animate-pulse delay-1000"></div>
        </div>
        
        <Container size="4" className="relative pt-20 pb-16">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
              Decentralized GPU Network
              <br />
              for Neural Network Training
            </h1>
            <p className="text-xl text-white/70 mb-8 max-w-2xl mx-auto">
              Monetize your idle GPU resources or access affordable compute power for AI development through our secure, verified peer-to-peer network.
            </p>
            <Flex gap="4" justify="center" mb="8">
              <Button size="3" className="bg-blue-500 hover:bg-blue-600 transition">
                Rent Out Your GPU
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
              <Button size="3" variant="outline" className="border-white/20 hover:bg-white/10">
                Access Computing Power
              </Button>
            </Flex>
            
            <div className="w-full max-w-4xl mx-auto mt-12">
                <div className="absolute">
                <HeroIllustration />
              </div>
              <img src="/public/compunir.jpg" alt="Hero Illustration" className="w-full h-auto relative z-10" />
            </div>
          </div>
        </Container>
      </section>

      {/* Features Grid */}
      <section className="py-20 bg-[#2d2d44]">
        <Container>
          <Heading size="8" align="center">Core Features</Heading>
          <div className="mt-4">
          <Text size="4" color="gray" align="center" className="mb-12">
            Our decentralized network provides benefits for both GPU providers and AI developers.
          </Text>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pt-8">
            {[
              {
                icon: <Server className="h-6 w-6 text-blue-500" />,
                title: "GPU Resource Sharing",
                description: "Monetize idle GPU resources or access affordable compute power on demand."
              },
              {
                icon: <Shield className="h-6 w-6 text-green-500" />,
                title: "Verified Computation",
                description: "Multi-node consensus ensures computational integrity and accurate results."
              },
              {
                icon: <Lock className="h-6 w-6 text-purple-500" />,
                title: "Secure Payments",
                description: "Private Monero-based transactions with escrow protection."
              },
              {
                icon: <Cpu className="h-6 w-6 text-red-500" />,
                title: "Flexible Participation",
                description: "Set your own hardware limits and availability schedule."
              },
              {
                icon: <BarChart className="h-6 w-6 text-yellow-500" />,
                title: "Real-time Monitoring",
                description: "Track performance, earnings, and network status through our dashboard."
              },
              {
                icon: <Network className="h-6 w-6 text-indigo-500" />,
                title: "Sybil Protection",
                description: "Advanced protection against network manipulation through PoW and stake."
              }
            ].map((feature, index) => (
              <Card key={index} className="p-6">
                <Flex direction="column" gap="3">
                  <div className="p-3 rounded-lg bg-slate-50 w-fit">
                    {feature.icon}
                  </div>
                  <Heading size="4">{feature.title}</Heading>
                  <Text color="gray">{feature.description}</Text>
                </Flex>
              </Card>
            ))}
          </div>
        </Container>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <Container>
          <Card className="p-12 text-center">
            <Heading size="7">Ready to Join the Network?</Heading>
            <div className="mt-4">
            <Text size="4" color="gray" className="mb-8 max-w-xl mx-auto pb-8">
              Start monetizing your idle GPU resources or access affordable computing power for your AI projects today.
            </Text>
            </div>
            <div className="pt-8">
                <Flex gap="4" justify="center">
                <Button size="4">
                    Get Started as Provider
                    <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
                <Button size="4" variant="outline">
                    Develop with Compunir
                </Button>
                </Flex>
            </div>
          </Card>
        </Container>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-white py-12">
        <Container>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
            <div>
              <Heading size="4" className="mb-4">Compunir</Heading>
              <div className="mt-4">
              <Text size="2" color="gray">
                A decentralized GPU network for distributed machine learning training.
              </Text>
              </div>
            </div>
            <div>
              <Heading size="2" className="mb-4">Resources</Heading>
              <div className="mt-4">
              <Flex direction="column" gap="2">
                <Button variant="ghost" size="1">Documentation</Button>
                <Button variant="ghost" size="1">API Reference</Button>
                <Button variant="ghost" size="1">GitHub</Button>
              </Flex>
              </div>
            </div>
            <div>
              <Heading size="2" className="mb-4">Company</Heading>
              <div className="mt-4">
              <Flex direction="column" gap="2">
                <Button variant="ghost" size="1">About</Button>
                <Button variant="ghost" size="1">Blog</Button>
                <Button variant="ghost" size="1">Contact</Button>
              </Flex>
              </div>
            </div>
            <div>
              <Heading size="2" className="mb-4">Legal</Heading>
              <div className="mt-4">
              <Flex direction="column" gap="2">
                <Button variant="ghost" size="1">Terms</Button>
                <Button variant="ghost" size="1">Privacy</Button>
                <Button variant="ghost" size="1">Security</Button>
              </Flex>
              </div>
            </div>
          </div>
          <Separator className="my-8" />
          <Text size="1" color="gray" align="center">
            © 2025 Compunir. All rights reserved.
          </Text>
        </Container>
      </footer>
    </div>
  );
};

export default LandingPage;