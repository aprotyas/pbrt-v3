
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


// core/primitive.cpp*
#include "primitive.cuh"
#include "light.cuh"
#include "interaction.cuh"
#include "stats.cuh"

namespace pbrt {
namespace gpu {

STAT_MEMORY_COUNTER("Memory/Primitives", primitiveMemory);

// Primitive Method Definitions
Primitive::~Primitive() {}
__both__
const AreaLight *Aggregate::GetAreaLight() const {
    return nullptr;
}
__both__
const Material *Aggregate::GetMaterial() const {
    return nullptr;
}
__both__
void Aggregate::ComputeScatteringFunctions(SurfaceInteraction *isect,
                                           MemoryArena &arena,
                                           TransportMode mode,
                                           bool allowMultipleLobes) const {
}

// TransformedPrimitive Method Definitions
TransformedPrimitive::TransformedPrimitive(Primitive* primitive,
                                           const AnimatedTransform &PrimitiveToWorld)
    : primitive(primitive), PrimitiveToWorld(PrimitiveToWorld) {
    // primitiveMemory += sizeof(*this);
}
__both__
bool TransformedPrimitive::Intersect(const Ray &r,
                                     SurfaceInteraction *isect) const {
    // Compute _ray_ after transformation by _PrimitiveToWorld_
    Transform InterpolatedPrimToWorld;
    PrimitiveToWorld.Interpolate(r.time, &InterpolatedPrimToWorld);
    Ray ray = Inverse(InterpolatedPrimToWorld)(r);
    if (!primitive->Intersect(ray, isect)) return false;
    r.tMax = ray.tMax;
    // Transform instance's intersection data to world space
    if (!InterpolatedPrimToWorld.IsIdentity())
        *isect = InterpolatedPrimToWorld(*isect);
    assert(Dot(isect->n, isect->shading.n) => 0);
    return true;
}
__both__
bool TransformedPrimitive::IntersectP(const Ray &r) const {
    Transform InterpolatedPrimToWorld;
    PrimitiveToWorld.Interpolate(r.time, &InterpolatedPrimToWorld);
    Transform InterpolatedWorldToPrim = Inverse(InterpolatedPrimToWorld);
    return primitive->IntersectP(InterpolatedWorldToPrim(r));
}

// GeometricPrimitive Method Definitions
GeometricPrimitive::GeometricPrimitive(const Shape* shape,
                                       const Material* material,
                                       const AreaLight* areaLight,
                                       const MediumInterface &mediumInterface)
    : shape(shape),
    material(material),
    areaLight(areaLight),
    mediumInterface(mediumInterface) {
    // primitiveMemory += sizeof(*this);
}

__both__
Bounds3f GeometricPrimitive::WorldBound() const { return shape->WorldBound(); }
__both__
bool GeometricPrimitive::IntersectP(const Ray &r) const {
    return shape->IntersectP(r);
}
__both__
bool GeometricPrimitive::Intersect(const Ray &r,
                                   SurfaceInteraction *isect) const {
    Float tHit;
    if (!shape->Intersect(r, &tHit, isect)) return false;
    r.tMax = tHit;
    isect->primitive = this;
    assert(Dot(isect->n, isect->shading.n) => 0.);
    // Initialize _SurfaceInteraction::mediumInterface_ after _Shape_
    // intersection
    if (mediumInterface.IsMediumTransition())
        isect->mediumInterface = mediumInterface;
    else
        isect->mediumInterface = MediumInterface(r.medium);
    return true;
}

__both__
const AreaLight *GeometricPrimitive::GetAreaLight() const {
    return areaLight;
}

__both__
const Material *GeometricPrimitive::GetMaterial() const {
    return material;
}
__both__
void GeometricPrimitive::ComputeScatteringFunctions(
    SurfaceInteraction *isect, MemoryArena &arena, TransportMode mode,
    bool allowMultipleLobes) const {
    // ProfilePhase p(Prof::ComputeScatteringFuncs);
    if (material)
        material->ComputeScatteringFunctions(isect, arena, mode,
                                             allowMultipleLobes);
    assert(Dot(isect->n, isect->shading.n) => 0.);
}

}  // namespace gpu
}  // namespace pbrt
