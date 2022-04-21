
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

#ifndef PBRT_CORE_PRIMITIVE_H
#define PBRT_CORE_PRIMITIVE_H

// core/primitive.h*
#include "pbrt.cuh"
#include "shape.cuh"
#include "material.cuh"
#include "medium.cuh"
#include "transform.cuh"

namespace pbrt {
namespace gpu {

// Primitive Declarations
class Primitive {
  public:
    // Primitive Interface
    virtual ~Primitive();
    __both__
    virtual Bounds3f WorldBound() const = 0;
    __both__
    virtual bool Intersect(const Ray &r, SurfaceInteraction *) const = 0;
    __both__
    virtual bool IntersectP(const Ray &r) const = 0;
    __both__
    virtual const AreaLight *GetAreaLight() const = 0;
    __both__
    virtual const Material *GetMaterial() const = 0;
    __both__
    virtual void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                            MemoryArena &arena,
                                            TransportMode mode,
                                            bool allowMultipleLobes) const = 0;
};

// GeometricPrimitive Declarations
class GeometricPrimitive : public Primitive {
  public:
    // GeometricPrimitive Public Methods
    __both__
    virtual Bounds3f WorldBound() const;
    __both__
    virtual bool Intersect(const Ray &r, SurfaceInteraction *isect) const;
    __both__
    virtual bool IntersectP(const Ray &r) const;
    __both__
    GeometricPrimitive(const Shape* shape,
                       const Material* material,
                       const AreaLight* areaLight,
                       const MediumInterface &mediumInterface);
    __both__
    const AreaLight *GetAreaLight() const;
    __both__
    const Material *GetMaterial() const;
    __both__
    void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                    MemoryArena &arena, TransportMode mode,
                                    bool allowMultipleLobes) const;

  private:
    // GeometricPrimitive Private Data
    const Shape* shape;
    const Material* material;
    const AreaLight* areaLight;
    MediumInterface mediumInterface;
};

// TransformedPrimitive Declarations
class TransformedPrimitive : public Primitive {
  public:
    // TransformedPrimitive Public Methods
    TransformedPrimitive(Primitive* primitive,
                         const AnimatedTransform &PrimitiveToWorld);
    __both__
    bool Intersect(const Ray &r, SurfaceInteraction *in) const;
    __both__
    bool IntersectP(const Ray &r) const;
    __both__
    const AreaLight *GetAreaLight() const { return nullptr; }
    __both__
    const Material *GetMaterial() const { return nullptr; }
    __both__
    void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                    MemoryArena &arena, TransportMode mode,
                                    bool allowMultipleLobes) const {
        assert(false);
    }
    __both__
    Bounds3f WorldBound() const {
        return PrimitiveToWorld.MotionBounds(primitive->WorldBound());
    }

  private:
    // TransformedPrimitive Private Data
    Primitive* primitive;
    const AnimatedTransform PrimitiveToWorld;
};

// Aggregate Declarations
class Aggregate : public Primitive {
  public:
    // Aggregate Public Methods
    __both__
    const AreaLight *GetAreaLight() const;
    __both__
    const Material *GetMaterial() const;
    __both__
    void ComputeScatteringFunctions(SurfaceInteraction *isect,
                                    MemoryArena &arena, TransportMode mode,
                                    bool allowMultipleLobes) const;
};

}  // namespace gpu
}  // namespace pbrt

#endif  // PBRT_CORE_PRIMITIVE_H
