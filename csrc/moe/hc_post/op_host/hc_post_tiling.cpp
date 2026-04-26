/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file gather_selection_kv_cache_tiling.cpp
 * \brief
 */

#include <algorithm>

#include "hc_post_tiling.h"
#include "hc_post_tiling_arch35.h"

namespace optiling {
constexpr int64_t DEFAULT_DEAL_DPARAM = 2048;
constexpr int64_t MIN_DEAL_DPARAM = 16;

namespace {
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}

int64_t RoundUpFloatElements(int64_t num)
{
    constexpr int64_t blockSize = 32;
    return ((num + blockSize / static_cast<int64_t>(sizeof(float)) - 1) /
            (blockSize / static_cast<int64_t>(sizeof(float)))) *
           (blockSize / static_cast<int64_t>(sizeof(float)));
}

int64_t EstimateDSplitUbSize(int64_t dOnceDealing, int64_t hcParam)
{
    int64_t xQueSize = 2 * dOnceDealing * 4;
    int64_t residualQueSize = 2 * dOnceDealing * 4;
    int64_t postQueSize = 2 * hcParam * 4;
    int64_t combQueSize = 2 * hcParam * hcParam * 4;
    int64_t outQueSize = 2 * hcParam * dOnceDealing * 4;
    int64_t xCastSize = dOnceDealing * 4;
    int64_t residualCastSize = dOnceDealing * 4;
    int64_t outCastSize = hcParam * dOnceDealing * 4;
    int64_t postCastSize = hcParam * 4;
    int64_t combCastSize = hcParam * hcParam * 4;
    int64_t postBrcbSize = RoundUpFloatElements(hcParam) * 32;
    int64_t combBrcbSize = RoundUpFloatElements(hcParam) * 32;
    int64_t tempSumSize = hcParam * dOnceDealing * 4;
    return xQueSize + residualQueSize + postQueSize + combQueSize + outQueSize +
           xCastSize + residualCastSize + outCastSize + postCastSize + combCastSize +
           postBrcbSize + combBrcbSize + tempSumSize;
}
} // namespace

ge::graphStatus HcPostTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(
        coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(
        ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    ubBlockSize_ = 32;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::GetShapeInfo()
{
    OPS_ERR_IF(
        context_ == nullptr, OPS_LOG_E("HcPostTiling", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    if (GetInputShapeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // dtype校验
    if (GetInputDtypeInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::GetInputShapeInfo()
{
    auto xInput = context_->GetInputShape(INPUT_IDX_X);
    OPS_ERR_IF(xInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get xInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape xShape = xInput->GetStorageShape();
    size_t dimsN = xShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "xInput dim:%lu should be 3.", dimsN),
        return ge::GRAPH_FAILED);

    auto residualInput = context_->GetInputShape(INPUT_IDX_RESIDUAL);
    OPS_ERR_IF(residualInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape residualShape = residualInput->GetStorageShape();
    dimsN = residualShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST4),
        OPS_LOG_E(context_->GetNodeName(), "residualInput dim:%lu should be 4.", dimsN),
        return ge::GRAPH_FAILED);

    auto postInput = context_->GetInputShape(INPUT_IDX_POST);
    OPS_ERR_IF(postInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape postShape = postInput->GetStorageShape();
    dimsN = postShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST3),
        OPS_LOG_E(context_->GetNodeName(), "postInput dim:%lu should be 3.", dimsN),
        return ge::GRAPH_FAILED);

    auto combInput = context_->GetInputShape(INPUT_IDX_COMB);
    OPS_ERR_IF(combInput == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualInput nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape combShape = combInput->GetStorageShape();
    dimsN = combShape.GetDimNum();
    OPS_ERR_IF((dimsN != CONST4),
        OPS_LOG_E(context_->GetNodeName(), "combInput dim:%lu should be 4.", dimsN),
        return ge::GRAPH_FAILED);
    bParam_ = xShape.GetDim(DIM_INDEX_0);
    sParam_ = xShape.GetDim(DIM_INDEX_1);
    dParam_ = xShape.GetDim(DIM_INDEX_2);
    hcParam_ = residualShape.GetDim(DIM_INDEX_2);
    OPS_ERR_IF((bParam_ <= 0 || sParam_ <= 0 || dParam_ <= 0 || hcParam_ <= 0),
        OPS_LOG_E(context_->GetNodeName(), "b, s, d and hc should be positive, got b=%ld, s=%ld, d=%ld, hc=%ld.",
                  bParam_, sParam_, dParam_, hcParam_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((residualShape.GetDim(DIM_INDEX_0) != bParam_ ||
                residualShape.GetDim(DIM_INDEX_1) != sParam_ ||
                residualShape.GetDim(DIM_INDEX_3) != dParam_),
        OPS_LOG_E(context_->GetNodeName(), "residual shape should be [b, s, hc, d]."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((postShape.GetDim(DIM_INDEX_0) != bParam_ ||
                postShape.GetDim(DIM_INDEX_1) != sParam_ ||
                postShape.GetDim(DIM_INDEX_2) != hcParam_),
        OPS_LOG_E(context_->GetNodeName(), "post shape should be [b, s, hc]."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((combShape.GetDim(DIM_INDEX_0) != bParam_ ||
                combShape.GetDim(DIM_INDEX_1) != sParam_ ||
                combShape.GetDim(DIM_INDEX_2) != hcParam_ ||
                combShape.GetDim(DIM_INDEX_3) != hcParam_),
        OPS_LOG_E(context_->GetNodeName(), "comb shape should be [b, s, hc, hc]."),
        return ge::GRAPH_FAILED);
    tilingData_.set_bParam(bParam_);
    tilingData_.set_sParam(sParam_);
    tilingData_.set_dParam(dParam_);
    tilingData_.set_hcParam(hcParam_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::GetInputDtypeInfo()
{
    auto xDesc = context_->GetInputDesc(INPUT_IDX_X);
    OPS_ERR_IF(xDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get xDesc nullptr."),
        return ge::GRAPH_FAILED);
    auto xDtype = xDesc->GetDataType();
    OPS_ERR_IF(
        (xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT),
        OPS_LOG_E(context_->GetNodeName(), "xDtype is not supported."),
        return ge::GRAPH_FAILED);

    auto residualDesc = context_->GetInputDesc(INPUT_IDX_RESIDUAL);
    OPS_ERR_IF(residualDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get residualDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType residualDtype = residualDesc->GetDataType();
    OPS_ERR_IF(
        (residualDtype != xDtype),
        OPS_LOG_E(context_->GetNodeName(), "residualDtype is not equal to xDtype."),
        return ge::GRAPH_FAILED);

    auto postDesc = context_->GetInputDesc(INPUT_IDX_POST);
    OPS_ERR_IF(postDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get postDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType postDtype = postDesc->GetDataType();
    OPS_ERR_IF(
        (postDtype != ge::DT_FLOAT16 && postDtype != ge::DT_BF16 && postDtype != ge::DT_FLOAT),
        OPS_LOG_E(context_->GetNodeName(), "postDtype is not supported."),
        return ge::GRAPH_FAILED);

    auto combDesc = context_->GetInputDesc(INPUT_IDX_COMB);
    OPS_ERR_IF(combDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get combDesc nullptr."),
        return ge::GRAPH_FAILED);
    ge::DataType combDtype = combDesc->GetDataType();
    OPS_ERR_IF(
        (combDtype != postDtype),
        OPS_LOG_E(context_->GetNodeName(), "combDtype is not equal to postDtype."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::DoOpTiling()
{
    int64_t batchSize = bParam_ * sParam_;

    int64_t useCoreNum = batchSize < coreNum_ ? batchSize : coreNum_;
    int64_t batchOneCore = CeilDiv(batchSize, static_cast<int64_t>(useCoreNum));
    int64_t batchOneCoreTail = batchOneCore - 1;
    int64_t frontCore = batchSize - batchOneCoreTail * useCoreNum;
    tilingData_.set_usedCoreNum(useCoreNum);
    tilingData_.set_batchOneCore(batchOneCore);
    tilingData_.set_batchOneCoreTail(batchOneCoreTail);
    tilingData_.set_frontCore(frontCore);
    int64_t dOnceDealing = std::min(dParam_, DEFAULT_DEAL_DPARAM);
    while (dOnceDealing > MIN_DEAL_DPARAM &&
           EstimateDSplitUbSize(dOnceDealing, hcParam_) > ubSize_) {
        dOnceDealing = std::max(MIN_DEAL_DPARAM, DownAlign(dOnceDealing / 2, MIN_DEAL_DPARAM));
    }
    OPS_ERR_IF(EstimateDSplitUbSize(dOnceDealing, hcParam_) > ubSize_,
        OPS_LOG_E(context_->GetNodeName(),
                  "HcPost tiling failed: no available UB, ubSize=%ld, required=%ld, hc=%ld, dOnce=%ld, d=%ld",
                  ubSize_, EstimateDSplitUbSize(dOnceDealing, hcParam_), hcParam_, dOnceDealing, dParam_),
        return ge::GRAPH_FAILED);

    int64_t dSplitTime = dParam_ / dOnceDealing;
    tilingData_.set_dSplitTime(dSplitTime);
    tilingData_.set_dOnceDealing(dOnceDealing);
    tilingData_.set_dLastDealing(dParam_ % dOnceDealing);
    context_->SetBlockDim(useCoreNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::PostTiling()
{
    context_->SetTilingKey(0);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(16 * 1024 * 1024);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPostTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return PostTiling();
}

ge::graphStatus Tiling4HcPost(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForHcPost running.");
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForHcPost", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForHcPost", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        OPS_LOG_I(context, "Using arch35 tiling for ASCEND910_95");
        HcPostTilingRegbase tiling(context);
        return tiling.RunTilingRegbase();
    }
    HcPostTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4HcPost(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HcPost)
    .Tiling(Tiling4HcPost)
    .TilingParse<HcPostCompileInfo>(TilingPrepare4HcPost);

} // namespace optiling
