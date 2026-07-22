import { useEffect, useRef, useState } from 'react';

/**
 * Hook for 3D Perspective Tilt with Spring Physics on Card Hover
 */
export function use3DTilt<T extends HTMLElement = HTMLDivElement>(intensity = 8) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Check if user prefers reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return;
    }

    let requestID: number;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = el.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      const rotateX = ((y - centerY) / centerY) * -intensity;
      const rotateY = ((x - centerX) / centerX) * intensity;

      requestID = requestAnimationFrame(() => {
        el.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateY(-4px) scale(1.01)`;
      });
    };

    const handleMouseLeave = () => {
      requestID = requestAnimationFrame(() => {
        el.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px) scale(1)';
      });
    };

    el.addEventListener('mousemove', handleMouseMove);
    el.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      cancelAnimationFrame(requestID);
      el.removeEventListener('mousemove', handleMouseMove);
      el.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [intensity]);

  return ref;
}

/**
 * Animated Numeric Counter hook for stats counters
 */
export function useAnimatedCounter(targetValue: number, duration = 800) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTimestamp: number | null = null;
    const startValue = 0;

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      // Apple snappy spring easing approximation formula
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      const currentCount = Math.floor(startValue + easedProgress * (targetValue - startValue));
      setCount(currentCount);

      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        setCount(targetValue);
      }
    };

    window.requestAnimationFrame(step);
  }, [targetValue, duration]);

  return count;
}
