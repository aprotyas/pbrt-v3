
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTS_DIFFUSE_H
#define PBRT_LIGHTS_DIFFUSE_H

// lights/diffuse.h*
#include "pbrt.h"
#include "light.h"
#include "primitive.h"

namespace pbrt {
namespace cpu {

// DiffuseAreaLight Declarations
class DiffuseAreaLight : public AreaLight {
  public:
    // DiffuseAreaLight Public Methods
    DiffuseAreaLight(const Transform &LightToWorld,
                     const MediumInterface &mediumInterface, const Spectrum &Le,
                     int nSamples, const std::shared_ptr<Shape> &shape,
                     bool twoSided = false);
    Spectrum L(const Interaction &intr, const Vector3f &w) const {
        return (twoSided || Dot(intr.n, w) > 0) ? Lemit : Spectrum(0.f);
    }
    Spectrum Power() const;
    Spectrum Sample_Li(const Interaction &ref, const Point2f &u, Vector3f *wo,
                       Float *pdf, VisibilityTester *vis) const;
    Float Pdf_Li(const Interaction &, const Vector3f &) const;
    Spectrum Sample_Le(const Point2f &u1, const Point2f &u2, Float time,
                       Ray *ray, Normal3f *nLight, Float *pdfPos,
                       Float *pdfDir) const;
    void Pdf_Le(const Ray &, const Normal3f &, Float *pdfPos,
                Float *pdfDir) const;

  protected:
    // DiffuseAreaLight Protected Data
    const Spectrum Lemit;
    std::shared_ptr<Shape> shape;
    // Added after book publication: by default, DiffuseAreaLights still
    // only emit in the hemimsphere around the surface normal.  However,
    // this behavior can now be overridden to give emission on both sides.
    const bool twoSided;
    const Float area;
};

std::shared_ptr<AreaLight> CreateDiffuseAreaLight(
    const Transform &light2world, const Medium *medium,
    const ParamSet &paramSet, const std::shared_ptr<Shape> &shape);

}  // namespace cpu
}  // namespace pbrt

#endif  // PBRT_LIGHTS_DIFFUSE_H
