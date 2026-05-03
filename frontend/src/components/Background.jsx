import { useCallback } from "react";
import Particles from "@tsparticles/react";
import { loadFull } from "tsparticles";

export default function Background() {
  const particlesInit = useCallback(async (engine) => {
    await loadFull(engine);
  }, []);

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      options={{
        background: {
          color: "#0f172a",
        },
        particles: {
          number: { value: 50 },
          color: { value: "#ffffff" },
          opacity: { value: 0.1 },
          size: { value: 2 },
          move: {
            enable: true,
            speed: 1,
          },
          links: {
            enable: true,
            distance: 150,
            color: "#ffffff",
            opacity: 0.1,
          },
        },
      }}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        zIndex: -1,
      }}
    />
  );
}