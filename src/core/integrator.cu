
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// core/integrator.cpp*
#include "integrator.cuh"
#include "scene.cuh"
#include "interaction.cuh"
#include "sampling.cuh"
#include "parallel.cuh"
#include "film.cuh"
#include "sampler.cuh"
#include "integrator.cuh"
#include "progressreporter.cuh"
#include "camera.cuh"
#include "stats.cuh"

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// Integrator Method Definitions
Integrator::~Integrator() {}

// Integrator Utility Functions
__both__
Spectrum UniformSampleOneLight(const Interaction &it, const Scene &scene,
                               MemoryArena &arena, Sampler &sampler,
                               bool handleMedia, const Distribution1D *lightDistrib) {
    // ProfilePhase p(Prof::DirectLighting);
    // Randomly choose a single light to sample, _light_
    int nLights = scene.lights_size;
    if (nLights == 0) return Spectrum(0.f);
    int lightNum;
    Float lightPdf;
    if (lightDistrib) {
        lightNum = lightDistrib->SampleDiscrete(sampler.Get1D(), &lightPdf);
        if (lightPdf == 0) return Spectrum(0.f);
    } else {
        lightNum = min((int)(sampler.Get1D() * nLights), nLights - 1);
        lightPdf = Float(1) / nLights;
    }
    Light const *light = scene.lights[lightNum];
    Point2f uLight = sampler.Get2D();
    Point2f uScattering = sampler.Get2D();
    return EstimateDirect(it, uScattering, *light, uLight,
                          scene, sampler, arena, handleMedia) / lightPdf;
}

__both__
Spectrum EstimateDirect(const Interaction &it, const Point2f &uScattering,
                        const Light &light, const Point2f &uLight,
                        const Scene &scene, Sampler &sampler,
                        MemoryArena &arena, bool handleMedia, bool specular) {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0, scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
            scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
        }
        if (!f.IsBlack()) {
            // Compute effect of visibility for light source sample
              if (!visibility.Unoccluded(scene)) {
                Li = Spectrum(0.f);
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
                if (IsDeltaLight(light.flags))
                    Ld += f * Li / lightPdf;
                else {
                    Float weight =
                        PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    Ld += f * Li * weight / lightPdf;
                }
            }
        }
    }

    // Sample BSDF with multiple importance sampling
    if (!IsDeltaLight(light.flags)) {
        Spectrum f;
        bool sampledSpecular = false;
        if (it.IsSurfaceInteraction()) {
            // Sample scattered direction for surface interactions
            BxDFType sampledType;
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                     bsdfFlags, &sampledType);
            f *= AbsDot(wi, isect.shading.n);
            sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
        }
        if (!f.IsBlack() && scatteringPdf > 0) {
            // Account for light contributions along sampled direction _wi_
            Float weight = 1;
            if (!sampledSpecular) {
                lightPdf = light.Pdf_Li(it, wi);
                if (lightPdf == 0) return Ld;
                weight = PowerHeuristic(1, scatteringPdf, 1, lightPdf);
            }

            // Find intersection and compute transmittance
            SurfaceInteraction lightIsect;
            Ray ray = it.SpawnRay(wi);
            Spectrum Tr(1.f);
            bool foundSurfaceInteraction =
                handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr)
                            : scene.Intersect(ray, &lightIsect);

            // Add light contribution from material sampling
            Spectrum Li(0.f);
            if (foundSurfaceInteraction) {
                if (lightIsect.primitive->GetAreaLight() == &light)
                    Li = lightIsect.Le(-wi);
            } else
                Li = light.Le(ray);
            if (!Li.IsBlack()) Ld += f * Li * Tr * weight / scatteringPdf;
        }
    }
    return Ld;
}

// SamplerIntegrator Method Definitions
void SamplerIntegrator::Render(const Scene &scene) {
    Preprocess(scene, *sampler);
    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 16;
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
    ProgressReporter reporter(nTiles.x * nTiles.y, "Rendering");
    {
        for (int y = 0; y < nTiles.y; ++y) {
            for (int x = 0; x < nTiles.x; ++x) {
                Point2i tile(x, y);
                           
                // Render section of image corresponding to _tile_

                // Allocate _MemoryArena_ for tile
                MemoryArena arena;

                // Get sampler instance for tile
                int seed = tile.y * nTiles.x + tile.x;
                std::vector<Sampler*> tileSamplers;

                // Compute sample bounds for tile
                int x0 = sampleBounds.pMin.x + tile.x * tileSize;
                int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
                int y0 = sampleBounds.pMin.y + tile.y * tileSize;
                int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
                Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
                LOG(INFO) << "Starting image tile " << tileBounds;

                // Get _FilmTile_ for tile
                std::unique_ptr<FilmTile> filmTile =
                    camera->film->GetFilmTile(tileBounds);

                // Creating unified objects for calculations
                CameraSample* cameraSamples;
                RayDifferential* rays;
                Float* rayWeights;
                Spectrum* Ls;
                cudaMallocHost(&cameraSamples, sizeof(CameraSample) 
                                * sampler->samplesPerPixel);
                cudaMallocHost(&rays,          sizeof(RayDifferential) 
                                * sampler->samplesPerPixel); 
                cudaMallocHost(&rayWeights,    sizeof(Float) 
                                * sampler->samplesPerPixel); 
                cudaMallocHost(&Ls,            sizeof(Spectrum) 
                                * sampler->samplesPerPixel); 

                new(cameraSamples) CameraSample[sampler->samplesPerPixel];
                new(rays)         RayDifferential[sampler->samplesPerPixel];
                new(rayWeights)   Float[sampler->samplesPerPixel];
                new(Ls)           Spectrum[sampler->samplesPerPixel];

                // Loop over pixels in tile to render them
                for (Point2i pixel : tileBounds) {
                    if (!InsideExclusive(pixel, pixelBounds))
                        continue;

                    for(int64_t sampleNum = 0; 
                        sampleNum < sampler->samplesPerPixel; 
                        sampleNum++) {

                        tileSamplers.push_back(sampler->Clone(seed));
                        {
                            ProfilePhase pp(Prof::StartPixel);
                            tileSamplers[sampleNum]->StartPixel(pixel);
                        }
                        tileSamplers[sampleNum]->SetSampleNumber(sampleNum);
                    
                        // Initialize _CameraSample_ for current sample
                        cameraSamples[sampleNum] =
                            tileSamplers[sampleNum]->GetCameraSample(pixel);

                        // Generate camera ray for current sample
                        
                        rayWeights[sampleNum] =
                            camera->GenerateRayDifferential(cameraSamples[sampleNum], &rays[sampleNum]);
                        rays[sampleNum].ScaleDifferentials(
                            1 / std::sqrt((Float)tileSamplers[sampleNum]->samplesPerPixel));
                        ++nCameraRays;
                    }
                    
                    // Evaluate radiance along camera ray
                    LiKernel<<<1, sampler->samplesPerPixel>>>
                        (Ls, this, rays, rayWeights, scene, tileSamplers.data(), arena);
                    cudaDeviceSynchronize();

                    for(int64_t sampleNum = 0; 
                        sampleNum < sampler->samplesPerPixel; 
                        sampleNum++) {
                        
                        // Ls[sampleNum] = 0.f;
                        // if (rayWeights[sampleNum] > 0) 
                        //     Ls[sampleNum] = 
                        //         Li(rays[sampleNum], scene, *tileSamplers[sampleNum], arena);

                        // Issue warning if unexpected radiance value returned
                        if (Ls[sampleNum].HasNaNs()) {
                            LOG(ERROR) << StringPrintf(
                                "Not-a-number radiance value returned "
                                "for pixel (%d, %d), sample %d. Setting to black.",
                                pixel.x, pixel.y,
                                (int)tileSamplers[sampleNum]->CurrentSampleNumber());
                            Ls[sampleNum] = Spectrum(0.f);
                        } else if (Ls[sampleNum].y() < -1e-5) {
                            LOG(ERROR) << StringPrintf(
                                "Negative luminance value, %f, returned "
                                "for pixel (%d, %d), sample %d. Setting to black.",
                                Ls[sampleNum].y(), pixel.x, pixel.y,
                                (int)tileSamplers[sampleNum]->CurrentSampleNumber());
                            Ls[sampleNum] = Spectrum(0.f);
                        } else if (std::isinf(Ls[sampleNum].y())) {
                            LOG(ERROR) << StringPrintf(
                                "Infinite luminance value returned "
                                "for pixel (%d, %d), sample %d. Setting to black.",
                                pixel.x, pixel.y,
                                (int)tileSamplers[sampleNum]->CurrentSampleNumber());
                            Ls[sampleNum] = Spectrum(0.f);
                        }
                        VLOG(1) << "Camera sample: " << cameraSamples[sampleNum] 
                                << " -> ray: " << rays[sampleNum]
                                << " -> Ls[sampleNum] = " << Ls[sampleNum];

                        // Add camera ray's contribution to image
                        filmTile->AddSample(cameraSamples[sampleNum].pFilm, 
                                            Ls[sampleNum], rayWeights[sampleNum]);

                        // Free _MemoryArena_ memory from computing image sample
                        // value
                        arena.Reset();
                    }
                }
                LOG(INFO) << "Finished image tile " << tileBounds;

                // Merge image tile into _Film_
                camera->film->MergeFilmTile(std::move(filmTile));
                reporter.Update();
                cudaFree(cameraSamples);
                cudaFree(rays);
                cudaFree(rayWeights);
                cudaFree(Ls);
            }
        }
        reporter.Done();
    }
    LOG(INFO) << "Rendering finished";

    // Save final image after rendering
    camera->film->WriteImage();

}
__global__
void LiKernel(Spectrum* Ls, SamplerIntegrator* integrator,
              const RayDifferential* rays, const Float* rayWeights,
              const Scene &scene, Sampler** tileSamplers,
              MemoryArena &arena) {
    int sampleNum = threadIdx.x;
    Ls[sampleNum] = 0.f;
    if (rayWeights[sampleNum] > 0)
        Ls[sampleNum] = integrator->Li(rays[sampleNum], scene, *tileSamplers[sampleNum], arena);
}

}  // namespace pbrt
